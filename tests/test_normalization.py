"""
tests/test_normalization.py — Test Suite for Unified RF Certificate Normalization Engine
"""

import unittest
from datetime import date
from core.normalization import (
    NO_EXPIRY_DATE,
    clean_string_field,
    normalize_country,
    parse_date,
    format_date_iso,
    normalize_certificate_fields,
    normalize_certificate_dict,
)


class TestNormalizationEngine(unittest.TestCase):

    def test_clean_string_field(self):
        self.assertEqual(clean_string_field(""), "")
        self.assertEqual(clean_string_field(None), "")
        self.assertEqual(clean_string_field("  "), "")
        self.assertEqual(clean_string_field("—"), "")
        self.assertEqual(clean_string_field("N/A"), "")
        self.assertEqual(clean_string_field("n/a"), "")
        self.assertEqual(clean_string_field("Unknown"), "")
        self.assertEqual(clean_string_field("Not Found"), "")
        self.assertEqual(clean_string_field("Valid Text"), "Valid Text")

    def test_country_canonicalization_us(self):
        us_variants = ["USA", "usa", "U.S.A.", "U.S.", "US", "us", "United States of America", "United States"]
        for v in us_variants:
            self.assertEqual(normalize_country(v), "United States", f"Failed for {v}")

    def test_country_canonicalization_other_common(self):
        cases = {
            "UK": "United Kingdom",
            "uk": "United Kingdom",
            "U.K.": "United Kingdom",
            "Great Britain": "United Kingdom",
            "UAE": "United Arab Emirates",
            "u.a.e.": "United Arab Emirates",
            "South Korea": "South Korea",
            "Korea, Republic of": "South Korea",
            "Republic of Korea": "South Korea",
            "The Netherlands": "Netherlands",
            "Holland": "Netherlands",
            "Czechia": "Czech Republic",
            "Russia": "Russian Federation",
            "PRC": "China",
            "People's Republic of China": "China",
            "Cote d'Ivoire": "Ivory Coast",
            "Myanmar": "Union of Myanmar",
            "Burma": "Union of Myanmar",
            "DR Congo": "Rep. Dem. Congo",
            "Swaziland": "Eswatini (Swaziland)",
        }
        for raw, expected in cases.items():
            self.assertEqual(normalize_country(raw), expected, f"Failed for {raw}")

    def test_territory_preservation(self):
        self.assertEqual(normalize_country("Puerto Rico"), "Puerto Rico (USA)")
        self.assertEqual(normalize_country("Puerto Rico (USA)"), "Puerto Rico (USA)")
        self.assertEqual(normalize_country("Montserrat (UK)"), "Montserrat (UK)")

    def test_country_placeholders_return_empty(self):
        for ph in ["", "—", "N/A", "Unknown", "None", "null", "Not Found"]:
            self.assertEqual(normalize_country(ph), "")

    def test_date_parsing_standard(self):
        self.assertEqual(parse_date("2024-05-12"), date(2024, 5, 12))
        self.assertEqual(parse_date("12/05/2024"), date(2024, 5, 12))
        self.assertEqual(parse_date("12-May-2024"), date(2024, 5, 12))

    def test_date_parsing_perpetual_sentinels(self):
        self.assertEqual(parse_date("Perpetual"), NO_EXPIRY_DATE)
        self.assertEqual(parse_date("infinite"), NO_EXPIRY_DATE)
        self.assertEqual(parse_date("no expiry"), NO_EXPIRY_DATE)
        self.assertEqual(parse_date("2999-01-01"), NO_EXPIRY_DATE)
        self.assertEqual(parse_date("9999-01-01"), NO_EXPIRY_DATE)

    def test_date_formatting(self):
        self.assertEqual(format_date_iso(date(2024, 5, 12)), "2024-05-12")
        self.assertEqual(format_date_iso(NO_EXPIRY_DATE), "Infinite")
        self.assertIsNone(format_date_iso(None))

    def test_normalize_certificate_fields_in_isolation(self):
        norm, autofilled = normalize_certificate_fields(
            raw_component="VSM-125kHz",
            raw_supplier="Aptiv Services",
            raw_country="USA",
            raw_certif_number="H-22392",
            raw_authority="FCC",
            raw_issue_date="2023-01-15",
            raw_exp_date="Perpetual",
            raw_cert_link="http://example.com/cert.pdf",
            raw_file_name="cert.pdf",
            db=None,
        )

        self.assertEqual(norm["component"], "VSM-125kHz")
        self.assertEqual(norm["country"], "United States")
        self.assertEqual(norm["authority"], "FCC")
        self.assertEqual(norm["issue_date"], date(2023, 1, 15))
        self.assertEqual(norm["exp_date"], NO_EXPIRY_DATE)
        self.assertEqual(norm["exp_date_iso"], "Infinite")
        self.assertEqual(norm["cert_link"], "http://example.com/cert.pdf")
        self.assertEqual(norm["file_name"], "cert.pdf")

    def test_normalize_certificate_dict_empty_cell_invariance(self):
        payload = {
            "component": "—",
            "supplier": "N/A",
            "country": "Unknown",
            "certif_number": "null",
            "authority": "",
            "issue_date": "None",
            "exp_date": "—",
        }
        norm, autofilled = normalize_certificate_dict(payload, db=None)

        self.assertIsNone(norm["component"])
        self.assertIsNone(norm["supplier"])
        self.assertIsNone(norm["country"])
        self.assertIsNone(norm["certif_number"])
        self.assertIsNone(norm["authority"])
        self.assertIsNone(norm["issue_date"])
        self.assertIsNone(norm["exp_date"])


if __name__ == "__main__":
    unittest.main()
