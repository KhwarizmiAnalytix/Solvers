"""Include-What-You-Use aspect for the Solvers Bazel build.

Usage (from .bazelrc or command line):
  bazel build --config=iwyu //...

The aspect runs IWYU on every C++ source in targets it visits and emits a
<src>.iwyu.txt report file into the output group "iwyu_report".  Build errors
are suppressed (--error=0) so suggestions are advisory only.
"""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

IwyuInfo = provider(
    doc = "Propagates IWYU output files.",
    fields = {"report_files": "depset of .iwyu.txt report files"},
)

def _iwyu_aspect_impl(target, ctx):
    if CcInfo not in target:
        return [IwyuInfo(report_files = depset())]

    srcs = getattr(ctx.rule.attr, "srcs", [])
    compilation_ctx = target[CcInfo].compilation_context

    reports = []
    for src_target in srcs:
        for src in src_target.files.to_list():
            if not (src.path.endswith(".cpp") or src.path.endswith(".cc") or src.path.endswith(".cxx")):
                continue

            out = ctx.actions.declare_file(src.basename + ".iwyu.txt", sibling = src)

            args = ctx.actions.args()
            args.add("-std=c++17")
            args.add("-Xiwyu", "--cxx17ns")
            args.add("-Xiwyu", "--error=0")
            args.add("-Xiwyu", "--no_fwd_decls")
            args.add("-Xiwyu", "--quoted_includes_first")
            args.add("-Xiwyu", "--transitive_includes_only")
            args.add("-Xiwyu", "--max_line_length=120")
            args.add_all(compilation_ctx.defines.to_list(), format_each = "-D%s")
            args.add_all(compilation_ctx.local_defines.to_list(), format_each = "-D%s")
            args.add_all(compilation_ctx.includes.to_list(), format_each = "-I%s")
            args.add_all(compilation_ctx.quote_includes.to_list(), format_each = "-iquote%s")
            args.add_all(compilation_ctx.system_includes.to_list(), format_each = "-isystem%s")
            args.add(src)

            iwyu_bin = ctx.attr._iwyu_binary[BuildSettingInfo].value

            ctx.actions.run_shell(
                inputs = depset(
                    [src],
                    transitive = [compilation_ctx.headers],
                ),
                outputs = [out],
                command = "{bin} $@ 2>&1 | tee {out}; exit 0".format(
                    bin = iwyu_bin,
                    out = out.path,
                ),
                arguments = [args],
                mnemonic = "IWYU",
                progress_message = "IWYU: %s" % src.short_path,
            )
            reports.append(out)

    return [
        IwyuInfo(report_files = depset(reports)),
        OutputGroupInfo(iwyu_report = depset(reports)),
    ]

iwyu_aspect = aspect(
    implementation = _iwyu_aspect_impl,
    attr_aspects = ["deps"],
    attrs = {
        "_iwyu_binary": attr.label(
            default = Label("//tools/iwyu:iwyu_binary"),
            providers = [BuildSettingInfo],
        ),
    },
    doc = "Runs include-what-you-use on C++ targets and emits advisory reports.",
)
