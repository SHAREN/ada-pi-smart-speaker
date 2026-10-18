import unittest
from unittest.mock import patch

from backend.web_tools import WEB_TOOL_DECLARATIONS, _SEARCH_CACHE, _public_ip, web_read, web_search


class WebToolsTests(unittest.IsolatedAsyncioTestCase):
    def test_public_ip_filter(self):
        self.assertTrue(_public_ip("1.1.1.1"))
        self.assertFalse(_public_ip("127.0.0.1"))
        self.assertFalse(_public_ip("192.168.0.136"))
        self.assertFalse(_public_ip("169.254.169.254"))
        self.assertFalse(_public_ip("::1"))

    async def test_loopback_read_is_blocked_without_exception(self):
        result = await web_read("http://127.0.0.1:18765/api/voice/status")
        self.assertFalse(result["success"])
        self.assertIn("blocked", result["error"].lower())

    async def test_private_lan_read_is_blocked_without_exception(self):
        result = await web_read("http://192.168.0.136:8123")
        self.assertFalse(result["success"])
        self.assertIn("blocked", result["error"].lower())

    def test_only_search_and_read_are_exposed(self):
        self.assertEqual(
            {item["name"] for item in WEB_TOOL_DECLARATIONS},
            {"web_search", "web_read"},
        )

    async def test_search_cache_avoids_repeated_backend_call(self):
        _SEARCH_CACHE.clear()
        fake = ([{"title": "Result", "url": "https://example.com", "snippet": "Fresh"}], "fake")
        with patch("backend.web_tools._search_sync", return_value=fake) as mocked:
            first = await web_search("same query", 2, "text")
            second = await web_search("same query", 2, "text")
        self.assertEqual(mocked.call_count, 1)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(first["results"], second["results"])


if __name__ == "__main__":
    unittest.main()
