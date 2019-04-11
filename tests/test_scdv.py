from unittest import TestCase
from benchdata.amazon_google_products import AmazonGoogleData
from feature_extracttion.text import EmbeddingTransformer
from feature_extracttion.text import Scdv

TEST_EMB_FILE = "../../data/pretrained/lexvec.test.small.vectors"

# TODO replace wih nose2

class TestScdv(TestCase):
    def test_fit_transform(self):
        data = AmazonGoogleData()
        data.load()
        data.default_fillna()

        titles = data.amazon_df["title"]
        titles = titles.tolist()[:100]
        del data

        N_COMPONENTS = 5

        emb_transformer = EmbeddingTransformer(TEST_EMB_FILE)
        emb_func = emb_transformer.token_to_embbeding
        model = Scdv(n_components=N_COMPONENTS, word_emb_func=emb_func)

        transformed = model.fit_transform(titles)

        expected_scdv_size = N_COMPONENTS * emb_transformer.model.vector_size
        n_samples = len(titles)
        self.assertEqual((n_samples, expected_scdv_size), transformed.shape)
        self.assertGreater(n_samples * expected_scdv_size, transformed.nnz)