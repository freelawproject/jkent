"""Tests for ``can_playwright_wait`` (continuation.py).

The autowait loop's Playwright-compatibility gate: which selectors can
be handed to ``page.wait_for_selector``. Migrated from
``tests/parsing/test_selector_utils.py`` when the function moved into
``ContinuationExecutor``'s module, next to its only consumer.
"""

from jkent.driver.unified_driver.continuation import can_playwright_wait


class TestCanPlaywrightWait:
    """Tests for can_playwright_wait function."""

    def test_css_selectors_always_compatible(self):
        """CSS selectors should always be compatible with Playwright."""
        assert can_playwright_wait("div.content", "css") is True
        assert can_playwright_wait("#main-content", "css") is True
        assert can_playwright_wait("table > tr:first-child", "css") is True
        assert can_playwright_wait("a[href]", "css") is True

    def test_element_targeting_xpath_compatible(self):
        """XPath selectors targeting elements should be compatible."""
        assert can_playwright_wait("//div", "xpath") is True
        assert can_playwright_wait("//div[@class='content']", "xpath") is True
        assert can_playwright_wait("//table//tr", "xpath") is True
        assert can_playwright_wait("//a[@href]", "xpath") is True
        assert can_playwright_wait("(//div)[1]", "xpath") is True

    def test_text_node_xpath_incompatible(self):
        """XPath selectors ending with /text() should be incompatible."""
        assert can_playwright_wait("//div/text()", "xpath") is False
        assert (
            can_playwright_wait("//p[@class='title']/text()", "xpath") is False
        )
        assert can_playwright_wait("//span//text()", "xpath") is False

    def test_attribute_xpath_incompatible(self):
        """XPath selectors targeting attributes should be incompatible."""
        assert can_playwright_wait("//a/@href", "xpath") is False
        assert can_playwright_wait("//div/@class", "xpath") is False
        assert can_playwright_wait("//input/@value", "xpath") is False
        assert can_playwright_wait("//table//td/@data-id", "xpath") is False

    def test_xpath_variable_references_incompatible(self):
        """Playwright can't bind XPath variable references ($var)."""
        assert can_playwright_wait("//div[@id=$section]", "xpath") is False
        assert can_playwright_wait("//a[position()=$n]", "xpath") is False

    def test_dollar_inside_string_literal_compatible(self):
        """A literal dollar sign in quoted text is not a variable."""
        assert can_playwright_wait("//a[text()='Price: $5']", "xpath") is True
        assert can_playwright_wait('//td[contains(., "US$")]', "xpath") is True

    def test_css_attribute_suffix_selector_compatible(self):
        """CSS [attr$=value] uses $ legitimately; css is always waitable."""
        assert can_playwright_wait("a[href$='.pdf']", "css") is True

    def test_exslt_functions_incompatible(self):
        """XPath selectors using EXSLT functions should be incompatible."""
        # Regular expression functions
        assert (
            can_playwright_wait("//div[re:test(., 'pattern')]", "xpath")
            is False
        )
        assert (
            can_playwright_wait("//div[re:match(@class, '\\d+')]", "xpath")
            is False
        )

        # String functions
        assert (
            can_playwright_wait("//div[str:concat('a', 'b')]", "xpath")
            is False
        )

        # Math functions
        assert can_playwright_wait("//div[math:max(1, 2)]", "xpath") is False

        # Set functions
        assert can_playwright_wait("//div[set:distinct(.)]", "xpath") is False

        # Dynamic functions
        assert (
            can_playwright_wait("//div[dyn:evaluate('.')]", "xpath") is False
        )

        # EXSL functions
        assert can_playwright_wait("//div[exsl:node-set(.)]", "xpath") is False

        # Function functions
        assert (
            can_playwright_wait("//div[func:function('name')]", "xpath")
            is False
        )

        # Date functions
        assert can_playwright_wait("//div[date:date-time()]", "xpath") is False

    def test_complex_element_xpath_compatible(self):
        """Complex XPath selectors targeting elements should be compatible."""
        # Predicates
        assert can_playwright_wait("//div[@id='main']", "xpath") is True
        assert can_playwright_wait("//tr[position() > 1]", "xpath") is True
        assert (
            can_playwright_wait("//div[contains(@class, 'active')]", "xpath")
            is True
        )

        # Nested paths
        assert (
            can_playwright_wait(
                "//div[@class='container']//table//tr", "xpath"
            )
            is True
        )

        # Multiple conditions
        assert (
            can_playwright_wait("//a[@href and @target='_blank']", "xpath")
            is True
        )

    def test_xpath_with_attribute_in_predicate_compatible(self):
        """XPath with @attribute in predicate (not at end) should be compatible."""
        # @href in predicate, not selecting the attribute
        assert can_playwright_wait("//a[@href]", "xpath") is True
        assert can_playwright_wait("//div[@class='test']", "xpath") is True
        assert (
            can_playwright_wait("//input[@type='text'][@name]", "xpath")
            is True
        )

    def test_whitespace_handling(self):
        """Function should handle selectors with leading/trailing whitespace."""
        assert can_playwright_wait("  //div  ", "xpath") is True
        assert can_playwright_wait("  //div/text()  ", "xpath") is False
        assert can_playwright_wait("  div.content  ", "css") is True

    def test_mixed_cases(self):
        """Test various edge cases and mixed selectors."""
        # Element with complex predicate - compatible
        assert (
            can_playwright_wait(
                "//div[contains(@class, 'item') and not(@disabled)]", "xpath"
            )
            is True
        )

        # Selecting parent element - compatible
        assert (
            can_playwright_wait("//div[@class='child']/parent::*", "xpath")
            is True
        )

        # Following sibling - compatible
        assert (
            can_playwright_wait(
                "//div[@id='first']/following-sibling::div", "xpath"
            )
            is True
        )

    def test_case_sensitivity(self):
        """EXSLT namespace prefixes are case-sensitive."""
        # Lowercase (standard) - incompatible
        assert can_playwright_wait("//div[re:test(., 'x')]", "xpath") is False

        # Uppercase - compatible (not recognized as EXSLT)
        assert can_playwright_wait("//div[RE:test(., 'x')]", "xpath") is True

    def test_attribute_in_middle_of_path(self):
        """Attribute in middle of path should still be considered compatible."""
        # This is unusual but the selector targets an element (the div)
        # Even though it has @id in the middle, it's not selecting @id at the end
        assert can_playwright_wait("//div[@id]/span", "xpath") is True
