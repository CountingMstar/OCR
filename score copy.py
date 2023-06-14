import pickle

###############
answers = []
OCR_answers = []
LLM_answers = []
###############


file_path0 = "data/data_sents.pickle" 
with open(file_path0, "rb") as file:
    data_sents = pickle.load(file)


file_path1 = "data/data.pickle" 
with open(file_path1, "rb") as file:
    data = pickle.load(file)


file_path2 = "data/OCR_results.pickle" 
with open(file_path2, "rb") as file:
    OCR_results = pickle.load(file)


file_path3 = "data/LLM_results.pickle" 
with open(file_path3, "rb") as file:
    LLM_results = pickle.load(file)


numbers = []
for i in range(len(LLM_results)):
    number = LLM_results[i][0]
    numbers.append(number)
    order = data[number]
    answers.append(data_sents[number][order])
    OCR_answers.append(OCR_results[number][1:][0][order][1])

    tmp = []

    for j in range(len(LLM_results[i][1:][0])):
        tmp.append(LLM_results[i][1:][0][j]["token_str"])

    LLM_answers.append(tmp)


OCR_score = 0
LLM_score1 = 0
LLM_score2 = 0
for i in range(len(numbers)):

    if answers[i] == OCR_answers[i]:
        OCR_score += 1

    if answers[i] == LLM_answers[i][0]:
        LLM_score1 += 1

    if answers[i] in LLM_answers[i]:
        LLM_score2 += 1














print('========= Score ========')
print(OCR_score / len(numbers))
print(LLM_score1 / len(numbers))
print(LLM_score2 / len(numbers))

print(LLM_score1 / OCR_score)
print(LLM_score2 / OCR_score)

