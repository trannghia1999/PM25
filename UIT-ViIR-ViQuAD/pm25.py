  #coding=utf8

from re import purge
import math
import json

import pandas as pd
from sklearn.metrics import classification_report
texts = open('texts.json')
data_texts = json.load(texts)
retrieved_results = open('retrieved_results.json')
question = open('questions.json')
texts = open('texts.json')


data_retrieved_results = json.load(retrieved_results)
data_question = json.load(question)
data_texts = json.load(texts)


question = open('questions.json')
data_question = json.load(question)

# print(data_question['uit_01__01154_0_1'])
# print(data_question['uit_01__01154_0_2'])
# print(data_question['uit_01__01154_0_3'])
# print(data_question['uit_01__01154_0_4'])
# print(data_question['uit_01__01154_0_5'])
# print('-----------')
# print(data_texts['uit_01__01154'])
question_demo = {
    "uit_01__01309_0_1":"Người mẹ khám thai ở nơi khác phát hiện được gì?",
    "uit_01__01309_0_2":"Hai bé bị dính nhau ở đâu? ",
    "uit_01__01309_0_3":"Sau vài ngày nhập viện, một bé có dấu hiệu gì? ",
}
ans_demo = {
    "a_0" : "Phó giáo sư Huỳnh Nguyễn Khánh Trang, Trưởng Khoa Sản bệnh, Bệnh viện Hùng Vương cho biết sản phụ mang thai con đầu lòng, nhập viện lúc thai 33 tuần tuổi. Người mẹ khám thai ở nơi khác và phát hiện song thai dính nhau. Các bác sĩ Bệnh viện Hùng Vương thực hiện chẩn đoán hình ảnh, thấy hai bé dính nhau ở phần bụng và cơ quan sinh dục. Đây là trường hợp song thai cùng trứng, có cùng bánh nhau và dây rốn. Sau vài ngày nhập viện, một bé có dấu hiệu thiếu máu nuôi, có thể gây nguy hiểm đến cả bé còn lại. Các bác sĩ hội chẩn và mổ sinh ngày 7/6, sớm 7 tuần so với dự kiến. Hai bé gái cân nặng 3,2 kg, được chuyển về Bệnh viện Nhi đồng Thành phố tiếp tục điều trị. Người mẹ hiện ổn định sức khỏe.",
    "a_1" : "Là chủ tiệm cơm ở Long An, thường xuyên phải tiếp xúc với ánh nắng mặt trời và nhiệt độ cao từ bếp lò. Chị không có thói quen chăm sóc da và sử dụng kem chống nắng khiến làn da lão hóa nhanh, vết nám, nếp nhăn và các gân máu nổi nhiều trên mặt. Các bác sĩ chẩn đoán chị bị lão hóa da sớm với làn da được đánh giá trên 35 tuổi do tiếp xúc trực tiếp với ánh nắng mặt trời và nhiệt độ cao. Sau một tháng điều trị theo phác đồ, tình trạng da chị được cải thiện, giảm đỏ và các gân máu dưới da. Bác sĩ Lê Thái Vân Thanh, Trưởng khoa Da liễu - Thẩm mỹ da, Bệnh viện Đại học Y Dược TP HCM, cho biết vào mùa nắng, người dân phải tiếp xúc và làm việc thường xuyên dưới thời tiết khắc nghiệt. Tia bức xạ mặt trời dễ dẫn đến những vấn đề về rối loạn tăng sắc tố da như nám da, đốm nâu, tàn nhang, quá trình lão hóa da diễn ra nhanh chóng. Nhiều người quan niệm chống lão hóa chỉ dành cho phụ nữ ở tuổi trung niên, khi các dấu hiệu đã được nhìn thấy rõ. Thực tế, quá trình lão hóa của cơ thể diễn ra từ khá sớm, ngay khi bước sang tuổi 20 đến 25. Từ tuổi 25 trở đi, mỗi năm làn da bị mất đi khoảng 1% lượng collagen. Độ đàn hồi của da suy giảm, nếp nhăn, tình trạng chảy xệ, sạm nám, đốm nâu xuất hiện nhiều, đặc biệt trên vùng da mặt, bác sĩ Thanh nói. Theo bác sĩ Thanh, nên bôi kem chống nắng đúng cách, uống nhiều nước, duy trì chế độ ăn giàu rau xanh và trái cây, ngủ đủ giấc, giảm stress, nâng cao sức khỏe thể chất và tinh thần với hoạt động thể dục thể thao, không hút thuốc và uống rượu bia... Khi ra ngoài đường từ 10h đến 16h, cần đeo khẩu trang, kính râm, mặc áo chống nắng, đội mũ rộng vành. Lưu ý chọn loại áo chống nắng có chất liệu thoáng mát, vải dày dặn, áo rộng sẽ có hiệu quả chống nắng tốt hơn. Chọn mỹ phẩm làm sạch da và dưỡng da an toàn hiệu quả, tránh sử dụng sản phẩm không rõ nguồn gốc khiến hàng rào bảo vệ da bị phá hủy. Khi có vấn đề về da, nên đến cơ sở y tế có các chuyên gia da liễu uy tín để tư vấn và lập phác đồ điều trị.",
}
demo = []
queries=[]
documents=[]
scores = []
for i in data_texts :
    demo.append(data_texts[i])

""" loại bỏ từ k cần thiết """
stopwords = set({})

# """ lọc từ """
texts = [
    [word  for word in document.lower().split() if word not in stopwords]
    for document in demo
]

word_count_dict = {}

for text in texts:
    for token in text:
        word_count = word_count_dict.get(token, 0) + 1
        word_count_dict[token] = word_count
texts = [[token for token in text if word_count_dict[token] > 3] for text in texts]
class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1

    def fit(self, corpus):
        tf = []
        df = {}
        idf = {}
        doc_len = []
        corpus_size = 0
        for document in corpus:
            corpus_size += 1
            doc_len.append(len(document))

            # compute tf (term frequency) per document
            frequencies = {}
            for term in document:
                term_count = frequencies.get(term, 0) + 1
                frequencies[term] = term_count

            tf.append(frequencies)

            # compute df (document frequency) per term
            for term, _ in frequencies.items():
                df_count = df.get(term, 0) + 1
                df[term] = df_count

        for term, freq in df.items():
            idf[term] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))

        self.tf_ = tf
        self.df_ = df
        self.idf_ = idf
        self.doc_len_ = doc_len
        self.corpus_ = corpus
        self.corpus_size_ = corpus_size
        self.avg_doc_len_ = sum(doc_len) / corpus_size
        return self

    def search(self, query):
        scores = [self._score(query, index) for index in range(self.corpus_size_)]
        return scores

    def _score(self, query, index):
        score = 0.0

        doc_len = self.doc_len_[index]
        frequencies = self.tf_[index]
        for term in query:
            if term not in frequencies:
                continue

            freq = frequencies[term]
            numerator = self.idf_[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len_)
            score += (numerator / denominator)

        return score

result = {}
for item in data_question :
    query = data_question[item]
    query = [word for word in query.lower().split() if word not in stopwords]
    bm25 = BM25()
    bm25.fit(texts)
    scores = bm25.search(query)
    max_score = 0
    maybe_doc = ''
    for score, doc in zip(scores, demo):
        score = round(score, 3)

        if score > max_score :
            max_score = score
            maybe_doc = doc
    for item_demo in data_texts :
                if data_texts[item_demo] == maybe_doc :
                    print(item + " : " + item_demo )
                    result[item] = item_demo
                    # with open('result_train.json', 'w') as f:
                    #  json.dump(result, f)
                    
                    
      
            
                   




