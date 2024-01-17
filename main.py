import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter

class BLEU:
    def __init__(self):
        self.__lower_n_split = lambda x: x.lower().split()
    
    def __make_ngrams(self, sentence, n):
        words = self.__lower_n_split(sentence)
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i : i + n]))
        return ngrams
    
    def __simple_precision(self, ca, refs, n):
        ngrams = self.__make_ngrams(ca, n)
        count = 0
        for ngram in ngrams:
            for ref in refs:
                if ngram in self.__make_ngrams(ref, n):
                    count += 1
                    break
        return count / len(ngrams)

    def __modified_precision(self, ca, refs, n):
        ngrams = self.__make_ngrams(ca, n)
        ngram_counts = Counter(ngrams)
        total_count = 0
        data = pd.DataFrame()
        words = []
        countPred = []
        countRef = []
        for ngram in set(ngrams):
            words.append(ngram)
            max_count = 0
            for ref in refs:
                max_count = max(max_count, Counter(self.__make_ngrams(ref, n)).get(ngram, 0))
            countPred.append(ngram_counts[ngram])
            countRef.append(max_count)
            total_count += min(max_count, ngram_counts[ngram])
        
        p = total_count / len(ngrams)
        if p != 0:
            data['Words'] = words
            data["Count in Prediction"] = countPred
            data["Count in Reference"] = countRef
            st.write(f"For {n}-gram")
            st.dataframe(data)
            st.write(f"∴ CLIPPED PRECISION = {p}")
            st.write("")
        return p
    
    def __brevity_penalty(self, ca, refs):
        ca_len = len(ca)
        if ca_len == 0:
            return 0
        cleaned_refs = (self.__lower_n_split(ref) for ref in refs)
        ref_lens = (len(ref) for ref in cleaned_refs)
        closest_ref_len = min(ref_lens, key = lambda ref_len: abs(ca_len - ref_len))
        bp = 1 if ca_len > closest_ref_len else np.exp(1 - closest_ref_len / ca_len)
        st.write(f"∴ BP = {bp}")
        return bp
    
    def bleu(self, ca, refs, n = None):
        n = len(ca.split()) if not n else n
        p_n = []
        n_ = 0
        latext = r'''
                $$ 
                    Clipped Precision = \frac{\sum Count In Reference}{\sum Count In Prediction} 
                $$ 
            '''
        st.write(latext)
        st.write("")
        st.markdown("#### CALCULATING CLIPPED PRECISION ...")
        for i in range(1, n + 1):
            p = self.__modified_precision(ca, refs, i)
            if p == 0:
                break
            p_n.append(p)
            n_ += 1
        
        st.write(f"∴ Ngram Used = {n_}")

        weights = [round(1/n_, 3) for _ in range(n_)]
        GAP = 1
        for w, p in zip(weights, p_n):
            GAP *= p**w
        
        
        st.markdown("#### CALCULATING GLOBAL AVERAGE PRECISION ...")
        latex = r'''
                $$
                    GAP = \prod {p ^ w} , \;\; w = \frac{1}{Ngrams \: used}
                $$
                '''
        st.write(latex)
        st.write(f"∴ GAP = {round(GAP, 3)}")
        st.write("")
        st.markdown("#### CALCULATING BREVITY PENALTY ...")
        latext = r'''
                $$
                    if \: Length of Predicted Text \leq Length of Reference Text 
                    \\
                    Brevity Penalty = e^{1 - \frac{Length of Predicted Text}{Length of Reference Text}}

                $$ 
            '''
        st.write(latext)
        latext = r'''
                $$
                    else \: Brevity Penalty = 1
                $$  
            '''
        st.write(latext)
        st.write("")
        BP = self.__brevity_penalty(ca, refs)
        st.write()
        latex = r'''
                $$
                    BLEU Score = GAP * BP
                $$
                '''
        st.write(latex)
        st.write("")
        return round(BP * GAP, 3)

st.set_page_config(page_title = "BLEU Score Calculator", layout = "wide")

print("==========START============")
st.title("BLEU Score Calculator")
st.markdown(
    "Made By: Umang Kirit Lodaya [GitHub](https://github.com/Umang-Lodaya/BLEU-Score-Calculator) | [LinkedIn](https://www.linkedin.com/in/umang-lodaya-074496242/) | [Kaggle](https://www.kaggle.com/umanglodaya)"
)

pred = ""; refs = ""

col1, col2 = st.columns([3, 2], gap = "medium")
with col1:
    # pred = st.text_input("Prediction Text", placeholder = "Enter your prediction text here")
    pred = st.text_input("Prediction Text", value = "It It It is raining heavily")
    # st.markdown(f"##### " + pred)
    # refs = st.text_area("Reference Text(s)", placeholder = "Enter your reference text here, each one on line")
    refs = st.text_area("Reference Text(s)", value = "It was a rainy day\nIt was raining heavily today")
    refs = refs.split("\n")

with col2:
    if pred and refs:
        st.dataframe(pd.DataFrame([pred], columns = ["Prediction Text"]), use_container_width = True)
        st.dataframe(pd.DataFrame(refs, columns = ["Reference Text"]), use_container_width = True)

start = st.button("Start Calculating")
st.write("")
if start:
    try:
        BL = BLEU()
        st.write("")
        
        st.markdown(f"### ∴ BLEU SCORE = {BL.bleu(pred, refs)}")

    except Exception as e:
        st.error(e)
