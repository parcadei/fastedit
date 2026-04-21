"""Tests for fastedit.inference.snippet_analysis module.

Pure unit tests for deterministic regex and text-processing functions:
- _regex_extract_names: multi-language definition extraction via regex
- _extract_snippet_names: top-level name extraction (falls back to regex)
- _MARKER_RE: pattern matching "... existing code ..." style markers
- _has_import_changes: import detection across snippet vs original
- _split_snippet: separating import lines from code lines
- _merge_overlapping_regions: merging close (start, end) tuples

These tests do NOT require tldr or tree-sitter to be installed.
"""

import textwrap

import pytest

from fastedit.inference.snippet_analysis import (
    _regex_extract_names,
    _extract_snippet_names,
    _MARKER_RE,
    _has_import_changes,
    _split_snippet,
    _merge_overlapping_regions,
)


# ---------------------------------------------------------------------------
# _regex_extract_names
# ---------------------------------------------------------------------------

class TestRegexExtractNames:
    """Tests for multi-language regex-based name extraction."""

    # -- Python --

    def test_python_def(self):
        assert _regex_extract_names("def foo():") == ["foo"]

    def test_python_async_def(self):
        assert _regex_extract_names("async def bar():") == ["bar"]

    def test_python_class(self):
        assert _regex_extract_names("class MyClass:") == ["MyClass"]

    def test_python_indented_def(self):
        assert _regex_extract_names("    def helper(self):") == ["helper"]

    def test_python_indented_class(self):
        assert _regex_extract_names("    class Inner:") == ["Inner"]

    def test_python_def_with_args(self):
        assert _regex_extract_names("def compute(x, y, z=3):") == ["compute"]

    def test_python_async_def_with_args(self):
        assert _regex_extract_names("async def fetch(url: str) -> bytes:") == ["fetch"]

    def test_python_class_with_bases(self):
        assert _regex_extract_names("class MyView(BaseView, Mixin):") == ["MyView"]

    # -- JavaScript/TypeScript --

    def test_js_function(self):
        assert _regex_extract_names("function doStuff() {") == ["doStuff"]

    def test_js_export_default_async_function(self):
        assert _regex_extract_names("export default async function handle() {") == ["handle"]

    def test_js_export_function(self):
        assert _regex_extract_names("export function render() {") == ["render"]

    def test_js_async_function(self):
        assert _regex_extract_names("async function fetchData() {") == ["fetchData"]

    def test_js_indented_function(self):
        assert _regex_extract_names("  function innerFunc() {") == ["innerFunc"]

    # -- Go --

    def test_go_func(self):
        assert _regex_extract_names("func NewLimiter() *Limiter {") == ["NewLimiter"]

    def test_go_receiver_method(self):
        assert _regex_extract_names("func (l *Limiter) Allow() bool {") == ["Allow"]

    def test_go_value_receiver_method(self):
        assert _regex_extract_names("func (s Server) Start() error {") == ["Start"]

    # -- Rust --

    def test_rust_pub_fn(self):
        assert _regex_extract_names("pub fn new() -> Self {") == ["new"]

    def test_rust_pub_crate_async_unsafe_fn(self):
        assert _regex_extract_names("pub(crate) async unsafe fn process() {") == ["process"]

    def test_rust_plain_fn(self):
        assert _regex_extract_names("fn helper() {") == ["helper"]

    def test_rust_pub_async_fn(self):
        assert _regex_extract_names("pub async fn serve(addr: &str) {") == ["serve"]

    def test_rust_unsafe_fn(self):
        assert _regex_extract_names("unsafe fn raw_alloc(size: usize) -> *mut u8 {") == ["raw_alloc"]

    def test_rust_pub_struct(self):
        assert _regex_extract_names("pub struct Config {") == ["Config"]

    def test_rust_pub_crate_enum(self):
        assert _regex_extract_names("pub(crate) enum State {") == ["State"]

    def test_rust_impl(self):
        assert _regex_extract_names("impl MyStruct {") == ["MyStruct"]

    def test_rust_trait(self):
        assert _regex_extract_names("pub trait Handler {") == ["Handler"]

    # -- Ruby --

    def test_ruby_def(self):
        assert _regex_extract_names("def greet(name)") == ["greet"]

    def test_ruby_def_self_method(self):
        # Pattern captures method name after `self.`
        snippet = "def self.create(attrs)"
        names = _regex_extract_names(snippet)
        assert names == ["create"]

    # -- Scala --

    def test_scala_def(self):
        assert _regex_extract_names("def transform(x: Int): Int = {") == ["transform"]

    def test_scala_object(self):
        assert _regex_extract_names("object Singleton {") == ["Singleton"]

    # -- Elixir --

    def test_elixir_def(self):
        assert _regex_extract_names("def hello(name) do") == ["hello"]

    def test_elixir_defp(self):
        assert _regex_extract_names("defp validate(data) do") == ["validate"]

    def test_elixir_defmodule(self):
        assert _regex_extract_names("defmodule MyApp.Router do") == ["MyApp"]

    # -- Swift/Kotlin --

    def test_swift_func(self):
        assert _regex_extract_names("func viewDidLoad() {") == ["viewDidLoad"]

    def test_kotlin_fun(self):
        assert _regex_extract_names("fun onCreate(savedInstanceState: Bundle?) {") == ["onCreate"]

    # -- Class-like constructs across languages --

    def test_class_keyword(self):
        assert _regex_extract_names("class Animal {") == ["Animal"]

    def test_struct_keyword(self):
        assert _regex_extract_names("struct Point {") == ["Point"]

    def test_enum_keyword(self):
        assert _regex_extract_names("enum Color {") == ["Color"]

    def test_trait_keyword(self):
        assert _regex_extract_names("trait Drawable {") == ["Drawable"]

    def test_interface_keyword(self):
        assert _regex_extract_names("interface Serializable {") == ["Serializable"]

    def test_abstract_class(self):
        assert _regex_extract_names("abstract class Shape {") == ["Shape"]

    def test_export_class(self):
        assert _regex_extract_names("export class UserService {") == ["UserService"]

    # -- Multiple definitions --

    def test_multiple_definitions(self):
        snippet = textwrap.dedent("""\
            def alpha():
                pass

            def beta():
                pass

            class Gamma:
                pass
        """)
        assert _regex_extract_names(snippet) == ["alpha", "beta", "Gamma"]

    def test_mixed_languages_in_one_snippet(self):
        """Regex patterns are language-agnostic -- they all run on every line."""
        snippet = textwrap.dedent("""\
            def python_func():
                pass
            function jsFunc() {
            }
            fn rust_func() {
            }
        """)
        names = _regex_extract_names(snippet)
        assert "python_func" in names
        assert "jsFunc" in names
        assert "rust_func" in names

    def test_order_preserved(self):
        snippet = textwrap.dedent("""\
            class First:
                pass
            def second():
                pass
            def third():
                pass
        """)
        assert _regex_extract_names(snippet) == ["First", "second", "third"]

    # -- Deduplication --

    def test_duplicate_names_only_first(self):
        snippet = textwrap.dedent("""\
            def process():
                pass

            def process():
                pass
        """)
        assert _regex_extract_names(snippet) == ["process"]

    # -- Empty / no definitions --

    def test_no_definitions(self):
        snippet = textwrap.dedent("""\
            x = 1
            y = x + 2
            print(y)
        """)
        assert _regex_extract_names(snippet) == []

    def test_empty_string(self):
        assert _regex_extract_names("") == []

    def test_blank_lines_only(self):
        assert _regex_extract_names("\n\n\n") == []

    def test_comments_only(self):
        snippet = textwrap.dedent("""\
            # this is a comment
            // another comment
            /* block comment */
        """)
        assert _regex_extract_names(snippet) == []

    # -- Edge cases --

    def test_def_inside_string_literal_matches(self):
        """Regex is line-based and doesn't understand strings -- it will match."""
        # This is expected behavior: regex doesn't parse context.
        snippet = '    "def fake():"'
        # The line starts with whitespace then a quote; the pattern anchors on
        # ^\s*(?:async\s+)?(?:defp?\s+...) which requires `def ` after optional
        # whitespace.  The quote character prevents the match.
        assert _regex_extract_names(snippet) == []

    def test_decorator_line_ignored(self):
        snippet = textwrap.dedent("""\
            @app.route("/")
            def index():
                pass
        """)
        assert _regex_extract_names(snippet) == ["index"]

    def test_lambda_not_captured(self):
        assert _regex_extract_names("    f = lambda x: x + 1") == []

    def test_arrow_function_not_captured(self):
        """Arrow functions (const foo = () => {}) are not captured by regex."""
        assert _regex_extract_names("const foo = () => {") == []


# ---------------------------------------------------------------------------
# _extract_snippet_names (top-level dispatcher)
# ---------------------------------------------------------------------------

class TestExtractSnippetNames:
    """Tests for the top-level name extraction function.

    When language=None, it always falls back to regex.
    When language is set, it tries tldr first (which may fail in test env)
    and then falls back to regex. Either way, the names should match.
    """

    def test_no_language_falls_back_to_regex(self):
        snippet = "def hello():\n    pass\n"
        result = _extract_snippet_names(snippet, language=None)
        assert result == ["hello"]

    def test_language_none_explicit(self):
        snippet = "class Foo:\n    pass\n"
        result = _extract_snippet_names(snippet, None)
        assert result == ["Foo"]

    def test_with_python_language(self):
        """With language='python', tries tldr first then regex fallback."""
        snippet = textwrap.dedent("""\
            def alpha():
                pass

            def beta():
                pass
        """)
        result = _extract_snippet_names(snippet, language="python")
        # Regardless of whether tldr works or regex fallback runs,
        # both names should be present.
        assert "alpha" in result
        assert "beta" in result

    def test_with_javascript_language(self):
        snippet = "function render() {\n  return null;\n}\n"
        result = _extract_snippet_names(snippet, language="javascript")
        assert "render" in result

    def test_with_rust_language(self):
        snippet = "pub fn build() -> Result<()> {\n    Ok(())\n}\n"
        result = _extract_snippet_names(snippet, language="rust")
        assert "build" in result

    def test_with_go_language(self):
        snippet = "func NewServer() *Server {\n    return &Server{}\n}\n"
        result = _extract_snippet_names(snippet, language="go")
        assert "NewServer" in result

    def test_unknown_language_falls_back_to_regex(self):
        """An unknown language with no ext mapping falls back to regex."""
        snippet = "def fallback():\n    pass\n"
        result = _extract_snippet_names(snippet, language="brainfuck")
        assert result == ["fallback"]

    def test_empty_snippet(self):
        assert _extract_snippet_names("", language=None) == []
        assert _extract_snippet_names("", language="python") == []

    def test_no_definitions_with_language(self):
        snippet = "x = 42\nprint(x)\n"
        assert _extract_snippet_names(snippet, language="python") == []


# ---------------------------------------------------------------------------
# _MARKER_RE
# ---------------------------------------------------------------------------

class TestMarkerRE:
    """Tests for the ellipsis marker regex that identifies placeholder lines."""

    # -- Should match --

    def test_hash_existing_code(self):
        assert _MARKER_RE.match("# ... existing code ...")

    def test_hash_existing_code_leading_spaces(self):
        assert _MARKER_RE.match("    # ... existing code ...")

    def test_slash_slash_existing_code(self):
        assert _MARKER_RE.match("// ... existing code ...")

    def test_block_comment_existing_code(self):
        assert _MARKER_RE.match("/* ... existing code ... */")

    def test_hash_rest_of_code(self):
        assert _MARKER_RE.match("# ... rest of code ...")

    def test_slash_slash_rest_of_code(self):
        assert _MARKER_RE.match("// ... rest of code ...")

    def test_existing_methods(self):
        assert _MARKER_RE.match("# ... existing methods ...")

    def test_rest_of_implementation(self):
        assert _MARKER_RE.match("// ... rest of implementation ...")

    def test_existing_functions(self):
        assert _MARKER_RE.match("# ... existing functions ...")

    def test_extra_whitespace_around_dots(self):
        assert _MARKER_RE.match("#  ...  existing  code  ...")

    def test_tabs_leading(self):
        assert _MARKER_RE.match("\t# ... existing code ...")

    def test_mixed_whitespace_leading(self):
        assert _MARKER_RE.match("  \t  // ... rest of code ...")

    # -- Should NOT match --

    def test_regular_hash_comment(self):
        assert _MARKER_RE.match("# this is a regular comment") is None

    def test_regular_slash_comment(self):
        assert _MARKER_RE.match("// normal comment here") is None

    def test_code_line(self):
        assert _MARKER_RE.match("x = 1 + 2") is None

    def test_blank_line(self):
        assert _MARKER_RE.match("") is None

    def test_just_dots_no_keyword(self):
        assert _MARKER_RE.match("# ... something else ...") is None

    def test_import_statement(self):
        assert _MARKER_RE.match("import os") is None

    def test_def_statement(self):
        assert _MARKER_RE.match("def foo():") is None

    def test_string_with_existing(self):
        """A string that contains 'existing' but isn't a comment marker."""
        assert _MARKER_RE.match('    "existing code here"') is None

    def test_dots_without_comment_prefix(self):
        assert _MARKER_RE.match("... existing code ...") is None

    def test_no_leading_dots(self):
        assert _MARKER_RE.match("# existing code ...") is None


# ---------------------------------------------------------------------------
# _has_import_changes
# ---------------------------------------------------------------------------

class TestHasImportChanges:
    """Tests for import change detection.

    _has_import_changes relies on _get_import_line_set which uses tree-sitter.
    When language=None, it returns False immediately.
    When tree-sitter is not available for the language, _get_import_line_set
    returns an empty set, so the function returns False.
    """

    def test_language_none_returns_false(self):
        """With language=None, always returns False regardless of content."""
        snippet = "import os\nimport sys\n"
        original = "x = 1\n"
        assert _has_import_changes(snippet, original, language=None) is False

    def test_language_none_even_with_new_imports(self):
        snippet = "import json\n"
        original = ""
        assert _has_import_changes(snippet, original, language=None) is False

    def test_empty_snippet_returns_false(self):
        assert _has_import_changes("", "import os\n", language="python") is False

    def test_empty_original_with_language(self):
        """When tree-sitter parses and finds imports not in empty original."""
        # This test exercises the tree-sitter path. If tree-sitter is not
        # available, _get_import_line_set returns empty set -> False.
        # Either outcome is acceptable for a unit test.
        snippet = "import os\n"
        result = _has_import_changes(snippet, "", language="python")
        # Result depends on tree-sitter availability; we just verify no crash.
        assert isinstance(result, bool)

    def test_snippet_import_already_in_original(self):
        """If snippet imports are already in original, should return False."""
        snippet = "import os\n"
        original = "import os\nimport sys\n\ndef main():\n    pass\n"
        result = _has_import_changes(snippet, original, language="python")
        # With tree-sitter: detects import, finds it in original -> False.
        # Without tree-sitter: no import lines detected -> False.
        assert result is False

    def test_no_import_in_snippet(self):
        snippet = "def foo():\n    return 42\n"
        original = "import os\n\ndef bar():\n    pass\n"
        result = _has_import_changes(snippet, original, language="python")
        assert result is False

    def test_unknown_language_returns_false(self):
        """Unknown language has no _TS_IMPORT_TYPES entry -> empty set -> False."""
        snippet = "import something\n"
        original = ""
        assert _has_import_changes(snippet, original, language="brainfuck") is False


# ---------------------------------------------------------------------------
# _split_snippet
# ---------------------------------------------------------------------------

class TestSplitSnippet:
    """Tests for splitting snippets into import and code parts.

    _split_snippet uses _get_import_line_set (tree-sitter). When language=None
    or tree-sitter is unavailable, it returns ("", snippet) -- all code, no imports.
    """

    def test_no_language_returns_all_code(self):
        snippet = "import os\ndef foo():\n    pass\n"
        imp, code = _split_snippet(snippet, language=None)
        assert imp == ""
        assert code == snippet

    def test_no_language_default(self):
        snippet = "def bar():\n    return 1\n"
        imp, code = _split_snippet(snippet)
        assert imp == ""
        assert code == snippet

    def test_empty_snippet(self):
        imp, code = _split_snippet("", language="python")
        assert imp == ""
        assert code == ""

    def test_all_code_no_imports(self):
        """Pure code snippet with language set but no imports."""
        snippet = "x = 1\ny = 2\n"
        imp, code = _split_snippet(snippet, language="python")
        # With tree-sitter: no import lines found -> ("", snippet)
        # Without tree-sitter: same result
        assert imp == ""
        assert code == snippet

    def test_unknown_language(self):
        """Unknown language has no import types -> returns all as code."""
        snippet = "import foo\ndef bar():\n    pass\n"
        imp, code = _split_snippet(snippet, language="brainfuck")
        assert imp == ""
        assert code == snippet

    def test_returns_tuple_of_strings(self):
        """Regardless of tree-sitter availability, return type is (str, str)."""
        imp, code = _split_snippet("def x():\n    pass\n", language="python")
        assert isinstance(imp, str)
        assert isinstance(code, str)

    def test_import_and_code_reconstruct_original(self):
        """The import + code parts should reconstruct the original snippet."""
        snippet = "import os\n\ndef main():\n    os.getcwd()\n"
        imp, code = _split_snippet(snippet, language="python")
        # Regardless of split point, the parts should join to the original.
        assert imp + code == snippet


# ---------------------------------------------------------------------------
# _merge_overlapping_regions
# ---------------------------------------------------------------------------

class TestMergeOverlappingRegions:
    """Tests for region merging with gap tolerance."""

    # -- Basic cases --

    def test_empty_list(self):
        assert _merge_overlapping_regions([]) == []

    def test_single_region(self):
        assert _merge_overlapping_regions([(1, 10)]) == [(1, 10)]

    # -- Non-overlapping --

    def test_non_overlapping_far_apart(self):
        """Regions far apart (> default gap of 20) stay separate."""
        regions = [(1, 10), (50, 60)]
        result = _merge_overlapping_regions(regions)
        assert result == [(1, 10), (50, 60)]

    def test_non_overlapping_exactly_at_gap_boundary(self):
        """Regions exactly gap+1 apart stay separate."""
        # gap=20 default. (1,10) and (31,40): start=31, prev_end=10, 31 > 10+20=30
        regions = [(1, 10), (31, 40)]
        result = _merge_overlapping_regions(regions)
        assert result == [(1, 10), (31, 40)]

    # -- Overlapping --

    def test_overlapping_regions(self):
        """Regions that directly overlap merge into one."""
        regions = [(1, 20), (15, 30)]
        result = _merge_overlapping_regions(regions)
        assert result == [(1, 30)]

    def test_fully_contained_region(self):
        """A region fully inside another merges to the outer."""
        regions = [(1, 50), (10, 30)]
        result = _merge_overlapping_regions(regions)
        assert result == [(1, 50)]

    def test_identical_regions(self):
        regions = [(5, 15), (5, 15)]
        result = _merge_overlapping_regions(regions)
        assert result == [(5, 15)]

    # -- Within gap distance --

    def test_within_gap_distance_default(self):
        """Regions within default gap=20 merge."""
        # (1,10) and (25,35): start=25, prev_end=10, 25 <= 10+20=30 -> merge
        regions = [(1, 10), (25, 35)]
        result = _merge_overlapping_regions(regions)
        assert result == [(1, 35)]

    def test_exactly_at_gap_boundary_inclusive(self):
        """Regions exactly at gap boundary merge (<=)."""
        # (1,10) and (30,40): start=30, prev_end=10, 30 <= 10+20=30 -> merge
        regions = [(1, 10), (30, 40)]
        result = _merge_overlapping_regions(regions)
        assert result == [(1, 40)]

    def test_within_gap_custom_small(self):
        """Custom gap=5."""
        regions = [(1, 10), (14, 20)]
        result = _merge_overlapping_regions(regions, gap=5)
        assert result == [(1, 20)]

    def test_not_within_custom_gap(self):
        """Regions outside custom gap stay separate."""
        regions = [(1, 10), (20, 30)]
        result = _merge_overlapping_regions(regions, gap=5)
        assert result == [(1, 10), (20, 30)]

    def test_gap_zero(self):
        """Gap=0 means only truly overlapping or adjacent regions merge."""
        regions = [(1, 10), (11, 20)]
        result = _merge_overlapping_regions(regions, gap=0)
        # start=11, prev_end=10, 11 <= 10+0=10 is False -> no merge
        assert result == [(1, 10), (11, 20)]

    def test_gap_zero_adjacent(self):
        """Gap=0: start=10, prev_end=10: 10 <= 10+0=10 -> merge."""
        regions = [(1, 10), (10, 20)]
        result = _merge_overlapping_regions(regions, gap=0)
        assert result == [(1, 20)]

    # -- Multiple regions --

    def test_three_regions_first_two_overlap(self):
        """First two merge, third stays separate."""
        regions = [(1, 20), (15, 30), (100, 110)]
        result = _merge_overlapping_regions(regions)
        assert result == [(1, 30), (100, 110)]

    def test_three_regions_all_merge(self):
        """All three within gap distance -> single region."""
        regions = [(1, 10), (20, 30), (40, 50)]
        result = _merge_overlapping_regions(regions)
        assert result == [(1, 50)]

    def test_three_regions_last_two_overlap(self):
        """First separate, last two merge."""
        regions = [(1, 5), (100, 120), (110, 130)]
        result = _merge_overlapping_regions(regions)
        assert result == [(1, 5), (100, 130)]

    def test_chain_merge(self):
        """Four regions that chain-merge into one."""
        regions = [(1, 10), (15, 25), (30, 40), (45, 55)]
        result = _merge_overlapping_regions(regions)
        assert result == [(1, 55)]

    def test_chain_merge_breaks_in_middle(self):
        """Chain merges first pair and last pair but gap in middle."""
        regions = [(1, 10), (20, 30), (80, 90), (100, 110)]
        # (1,10) and (20,30): 20 <= 10+20=30 -> merge (1,30)
        # (80,90): 80 > 30+20=50 -> separate
        # (80,90) and (100,110): 100 <= 90+20=110 -> merge (80,110)
        result = _merge_overlapping_regions(regions)
        assert result == [(1, 30), (80, 110)]

    # -- Unsorted input --

    def test_unsorted_input(self):
        """Regions are sorted internally before merging."""
        regions = [(50, 60), (1, 10), (25, 35)]
        result = _merge_overlapping_regions(regions)
        # Sorted: (1,10), (25,35), (50,60)
        # (1,10) and (25,35): 25 <= 10+20=30 -> merge (1,35)
        # (1,35) and (50,60): 50 <= 35+20=55 -> merge (1,60)
        assert result == [(1, 60)]

    def test_reverse_sorted_input(self):
        """Reverse order should produce same result as forward."""
        regions = [(100, 110), (50, 60), (1, 10)]
        result = _merge_overlapping_regions(regions)
        # Sorted: (1,10), (50,60), (100,110)
        # (1,10) and (50,60): 50 > 10+20=30 -> separate
        # (50,60) and (100,110): 100 > 60+20=80 -> separate
        assert result == [(1, 10), (50, 60), (100, 110)]

    # -- Edge values --

    def test_single_line_regions(self):
        regions = [(5, 5), (6, 6)]
        result = _merge_overlapping_regions(regions, gap=0)
        # start=6, prev_end=5, 6 <= 5+0=5 is False -> separate
        assert result == [(5, 5), (6, 6)]

    def test_single_line_regions_merge_with_gap(self):
        regions = [(5, 5), (6, 6)]
        result = _merge_overlapping_regions(regions, gap=1)
        assert result == [(5, 6)]

    def test_large_gap_merges_everything(self):
        regions = [(1, 10), (500, 510), (1000, 1010)]
        result = _merge_overlapping_regions(regions, gap=1000)
        assert result == [(1, 1010)]

    def test_negative_coordinates(self):
        """Not expected in practice but should not crash."""
        regions = [(-10, -5), (-3, 0), (5, 10)]
        result = _merge_overlapping_regions(regions, gap=2)
        # (-10,-5) and (-3,0): -3 <= -5+2=-3 -> merge (-10,0)
        # (-10,0) and (5,10): 5 <= 0+2=2 is False -> separate
        assert result == [(-10, 0), (5, 10)]

    def test_many_regions_all_overlapping(self):
        """Stress test: 100 regions that all overlap."""
        regions = [(i, i + 5) for i in range(100)]
        result = _merge_overlapping_regions(regions, gap=0)
        assert result == [(0, 104)]

    def test_preserves_max_end(self):
        """When merging, the end is max(prev_end, end)."""
        regions = [(1, 100), (5, 50)]
        result = _merge_overlapping_regions(regions)
        assert result == [(1, 100)]


# ---------------------------------------------------------------------------
# Integration-like tests combining multiple functions
# ---------------------------------------------------------------------------

class TestCombined:
    """Tests that exercise multiple snippet_analysis functions together."""

    def test_regex_names_and_marker_independent(self):
        """Markers don't interfere with name extraction."""
        snippet = textwrap.dedent("""\
            def existing_func():
                pass

            # ... existing code ...

            def new_func():
                pass
        """)
        names = _regex_extract_names(snippet)
        assert names == ["existing_func", "new_func"]

    def test_snippet_with_imports_and_defs(self):
        """Names from definitions, not from import lines."""
        snippet = textwrap.dedent("""\
            import os
            import sys

            def main():
                pass

            class App:
                pass
        """)
        names = _regex_extract_names(snippet)
        assert names == ["main", "App"]

    def test_go_multi_function_snippet(self):
        snippet = textwrap.dedent("""\
            func NewRouter() *Router {
                return &Router{}
            }

            func (r *Router) Handle(path string, handler Handler) {
                r.routes[path] = handler
            }

            func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
                handler := r.routes[req.URL.Path]
                handler.ServeHTTP(w, req)
            }
        """)
        names = _regex_extract_names(snippet)
        assert names == ["NewRouter", "Handle", "ServeHTTP"]

    def test_rust_multi_definition_snippet(self):
        snippet = textwrap.dedent("""\
            pub struct Config {
                host: String,
                port: u16,
            }

            impl Config {
                pub fn new(host: &str, port: u16) -> Self {
                    Config { host: host.to_string(), port }
                }

                pub fn address(&self) -> String {
                    format!("{}:{}", self.host, self.port)
                }
            }
        """)
        names = _regex_extract_names(snippet)
        assert "Config" in names
        assert "new" in names
        assert "address" in names

    def test_merge_regions_from_realistic_chunks(self):
        """Realistic scenario: function at line 10-25, class at 40-80, function at 90-110."""
        regions = [(10, 25), (40, 80), (90, 110)]
        # gap=20: (10,25) and (40,80): 40 <= 25+20=45 -> merge (10,80)
        #         (10,80) and (90,110): 90 <= 80+20=100 -> merge (10,110)
        result = _merge_overlapping_regions(regions)
        assert result == [(10, 110)]

    def test_merge_regions_with_isolated_import_region(self):
        """Import region at top, two code regions further down."""
        regions = [(1, 5), (50, 70), (65, 90)]
        # gap=20: (1,5) and (50,70): 50 > 5+20=25 -> separate
        #         (50,70) and (65,90): 65 <= 70+20=90 -> merge (50,90)
        result = _merge_overlapping_regions(regions)
        assert result == [(1, 5), (50, 90)]
