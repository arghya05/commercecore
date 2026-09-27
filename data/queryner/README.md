# QueryNER normalized starter data

Source: https://huggingface.co/datasets/bltlab/queryner
Authors: Chester Palen-Michel, Lizzie Liang, Zhe Wu, Constantine Lignos.
Paper: QueryNER: Segmentation of E-commerce Queries, LREC-COLING 2024.
https://aclanthology.org/2024.lrec-main.1178/

The source dataset card declares CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/
The underlying queries originate in Amazon ESCI; retain its attribution and provenance too:
https://github.com/amazon-science/esci-data

Changes: native BIO tags converted to half-open token spans and stable local IDs;
tokens joined with single spaces for a clearly labeled reconstructed text field.
No semantic label correction, new intent labels, hard/soft labels or raw character
offset claims. This derived format was created for the CommerceCore research plan.
The original publication PDF has separate publication terms; those do not replace
the dataset card license. See the pinned revision and SHA-256 hashes in manifest.json.

Use train.jsonl for training, validation.jsonl for development, test.jsonl only for
final evaluation. QueryNER is derived from ESCI: check cross-source query overlap
before combining corpora. Structural checks do not prove semantic label accuracy.
These files are a segmentation seed, not a complete commerce multitask training set.
