import numpy as np
import MeCab
from scipy.sparse import csr_matrix
from nose2.tools import such
from shortscikit.feature_extracttion.text import JapaneseTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from janome.tokenizer import Tokenizer as JanomeTokenizer

with such.A('sample test') as it:

    with it.having('MeCab wakati only JapaneseTokenizer test group'):
        mecab_wakati_tokenizer = JapaneseTokenizer('MeCab')
        mecab_wakati_test_str = 'すもももももももものうち'
        mecab_wakati_test_tokenizer =  MeCab.Tagger('-Owakati')
        mecab_wakati_test_result = mecab_wakati_test_tokenizer.parse(mecab_wakati_test_str).split(' ')[:-1]
        mecab_wakati_count_vectorizer = CountVectorizer(tokenizer=mecab_wakati_tokenizer.fit_transform)
        mecab_wakati_vectorizer_test = ['すもももすももももももすもももももももももすももももものうち']
        mecab_wakati_result0 = mecab_wakati_count_vectorizer.fit_transform(mecab_wakati_vectorizer_test).sorted_indices()
        mecab_wakati_result1 = csr_matrix(([1, 4, 1, 7, 4],([0, 0, 0, 0, 0],[0, 1, 2, 3, 4])), shape=(1,5))
        @it.should('JapaneseTokenizer fit method')
        def mecab_wakati_test1(case):
            it.assertIsInstance(mecab_wakati_tokenizer.fit(mecab_wakati_test_str), JapaneseTokenizer)

        @it.should('JapaneseTokenizer transform method')
        def mecab_wakati_test2(case):
            it.assertListEqual(mecab_wakati_tokenizer.transform(mecab_wakati_test_str), mecab_wakati_test_result)

        @it.should('JapaneseTokenizer fit_transform method')
        def mecab_wakati_test3(case):
            it.assertListEqual(mecab_wakati_tokenizer.fit_transform(mecab_wakati_test_str), mecab_wakati_test_result)

        @it.should('CountVectorizer use JapaneseTokenizer')
        def mecab_wakati_test4(case):
            it.assertTrue(np.alltrue(mecab_wakati_result0.todense() == mecab_wakati_result1.todense()))
    
    with it.having('janome wakati only JapaneseTokenizer test group'):
        janome_wakati_tokenizer = JapaneseTokenizer('janome')
        janome_wakati_test_str = 'すもももももももものうち'
        janome_wakati_test_tokenizer = JanomeTokenizer()
        janome_wakati_test_result = janome_wakati_test_tokenizer.tokenize(janome_wakati_test_str,wakati=True)
        janome_wakati_test_count_vectorizer = CountVectorizer(tokenizer=janome_wakati_tokenizer.fit_transform)
        janome_wakati_vectorizer_test = ['すもももすももももももすもももももももももすももももものうち']
        janome_wakati_result0 = janome_wakati_test_count_vectorizer.fit_transform(janome_wakati_vectorizer_test).sorted_indices()
        janome_wakati_result1 = csr_matrix(([1, 4, 1, 7, 4],([0, 0, 0, 0, 0],[0, 1, 2, 3, 4])), shape=(1,5))

        @it.should('JapaneseTokenizer fit method')
        def janome_wakati__test1(case):
            it.assertIsInstance(janome_wakati_tokenizer.fit(janome_wakati_test_str), JapaneseTokenizer)

        @it.should('JapaneseTokenizer transform method')
        def janome_wakati__test2(case):
            it.assertListEqual(janome_wakati_tokenizer.transform(janome_wakati_test_str), janome_wakati_test_result)

        @it.should('JapaneseTokenizer fit_transform method')
        def janome_wakati__test3(case):
            it.assertListEqual(janome_wakati_tokenizer.fit_transform(janome_wakati_test_str), janome_wakati_test_result)

        @it.should('CountVectorizer use JapaneseTokenizer')
        def janome_wakati__test4(case):
            it.assertTrue(np.alltrue(janome_wakati_result0.todense() == janome_wakati_result1.todense()))

    it.createTests(globals())