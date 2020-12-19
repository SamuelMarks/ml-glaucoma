from os import path
from tempfile import gettempdir
from unittest import TestCase
from unittest import main as unittest_main

from ml_glaucoma.cli_options.logparser.utils import ParsedLine, parse_line
from ml_glaucoma.utils import namedtuple2dict


class TestLogparserUtils(TestCase):
    def test_parse_line(self):
        base = path.join(gettempdir(), "ml_glaucoma_models")

        self.assertEqual(
            *map(
                namedtuple2dict,
                (
                    parse_line(
                        path.join(
                            base, "dr_spoc_grad_and_no_grad_EfficientNetB2_epochs_250"
                        )
                    ),
                    ParsedLine(
                        dataset="dr_spoc_grad_and_no_grad",
                        epoch=0,
                        value=0,
                        epochs=250,
                        transfer="EfficientNetB2",
                        loss="BinaryCrossentropy",
                        optimizer="Adam",
                        optimizer_params={"lr": 1e-3},
                        base="transfer",
                    ),
                ),
            )
        )

        self.assertEqual(
            *map(
                namedtuple2dict,
                (
                    parse_line(
                        path.join(base, "dr_spoc_no_no_grad_EfficientNetB2_epochs_250")
                    ),
                    ParsedLine(
                        dataset="dr_spoc_no_no_grad",
                        epoch=0,
                        value=0,
                        epochs=250,
                        transfer="EfficientNetB2",
                        loss="BinaryCrossentropy",
                        optimizer="Adam",
                        optimizer_params={"lr": 1e-3},
                        base="transfer",
                    ),
                ),
            )
        )

        self.assertEqual(
            *map(
                namedtuple2dict,
                (
                    parse_line(path.join(base, "dr_spoc_EfficientNetB2_epochs_250")),
                    ParsedLine(
                        dataset="dr_spoc",
                        epoch=0,
                        value=0,
                        epochs=250,
                        transfer="EfficientNetB2",
                        loss="BinaryCrossentropy",
                        optimizer="Adam",
                        optimizer_params={"lr": 1e-3},
                        base="transfer",
                    ),
                ),
            )
        )


if __name__ == "__main__":
    unittest_main()

__all__ = ["TestLogparserUtils"]
