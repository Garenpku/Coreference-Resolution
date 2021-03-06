import json
from nltk.tokenize import wordpunct_tokenize
#from conceptnet_util import ConceptNet
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import pandas as pd
import copy
import pickle


def load_data(path):
    data = json.loads(open(path).read())
    dialogs = []
    docs_for_tfidf = []
    for line in data:
        dialog = line['dialog']
        utterances = []
        utt_linear = ""
        for utterance in dialog:
            text = utterance['text']
            tokenized = wordpunct_tokenize(text)
            utterances.append(tokenized)
            utt_linear += text + ' '
        dialogs.append(utterances)
        docs_for_tfidf.append(utt_linear)
    return dialogs, docs_for_tfidf


def get_tfidf_list(docs_for_tfidf):
    cv = CountVectorizer()
    wcv = cv.fit_transform(docs_for_tfidf)
    tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(wcv)
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
    ascend_result = df_idf.sort_values(by=['idf_weights'])
    return ascend_result


def get_concepts(data, cn, stopwords):
    cpt_per_utt = [[list(set([word for word in sentence if word in cn.cpt_dict and word not in stopwords])) for sentence in dialog] for dialog in data]
    cpt_linear = []
    for dialog in cpt_per_utt:
        cpt = []
        for sent in dialog:
            cpt.extend(sent)
        cpt_linear.append(cpt)
    return cpt_per_utt, cpt_linear


def get_topK_by_tfidf(cpt_per_utt, cpt_linear, tf_idf, K):
    result = []
    dialogs_linear = []
    for i, dialog in enumerate(cpt_linear):
        in_dict = [word for word in dialog if word in tf_idf]
        res = sorted(in_dict, key=lambda x: tf_idf[x], reverse=True)
        if len(res) > K:
            result.append(res[:K])
        else:
            result.append(res)
        for j, utt in enumerate(cpt_per_utt[i]):
            cpt_per_utt[i][j] = [word for word in cpt_per_utt[i][j] if word in result[-1]]
        dialog_linear = []
        for utt in cpt_per_utt[i]:
            dialog_linear.extend(utt)
        dialogs_linear.append(dialog_linear)
    return dialogs_linear


def adjacent_ratio(response, context, cn):
    cnt = 0
    for i in range(len(context)):
        (res, _) = cn.expand_list(context[i])
        for cpt in response[i]:
            if cpt in res:
                cnt += 1
    return cnt / sum([len(line) for line in response])


def expand_by_path(concepts, k, cn, vocab=None, stopword=None):
    expanded = []
    total = len(concepts)
    per_step = int(total / 20)
    for i in range(len(concepts)):
        expanded.append(cn.expand_list_by_path(concepts[i], k, vocab, stopword))
        if i % per_step == 0:
            print("{}% completed.".format(i * 100 / total))
    return expanded


def split_cpt(concept, k):
    cpt_ctx = []
    cpt_res = []
    for dialog in concept:
        ctx = []
        for i in range(len(dialog) - k):
            ctx.extend(dialog[i])
        cpt_ctx.append(ctx)
        cpt_res.append(dialog[len(dialog)-k])
    return cpt_ctx, cpt_res


def write_file(path, dialogs, concepts, expanded, cpt_res=None):
    with open(path, 'w') as f:
        for i in range(len(dialogs)):
            dialog_str = ""
            concept_str = ""
            for j in range(len(dialogs[i]) -1):
                dialog_str += " ".join(dialogs[i][j]) + " <eou> "
                concept_str += " ".join(concepts[i][j]) + " <eou> "
            concept_str += " <expand> " + " ".join(expanded[i])
            if cpt_res:
                concept_str += " <response> " + " ".join(cpt_res[i])
            concept_str += " <index> " + str(i)
            response = " ".join(dialogs[i][-1])
            f.write(dialog_str + '\t' + response + '\t' + concept_str + '\n')


def guide_rate_concept(cpt_res, cpt_ctx):
    count = 0
    tmp = [[word for word in cpt_res[i] if word in cpt_ctx[i]] for i in range(len(cpt_ctx))]
    return sum([len(line) for line in tmp]) / sum([len(line) for line in cpt_res])


# the ratio of responses that can be guided by the context
def guide_rate(cpt_res, cpt_ctx):
    count = 0
    for i in range(len(cpt_res)):
        for word in cpt_res[i]:
            if word in cpt_ctx[i]:
                count += 1
                break
    return count / len(cpt_res)


# the ratio of concepts that are related to the response, sorted by distance
def distance_rate(cpt_res, cpt_per_utt, cn):
    distance_dict = {}
    cpt_ctx = [line[:-1] for line in cpt_per_utt]
    for i in range(len(cpt_res)):
        candidates, _ = cn.expand_list(cpt_res[i])
        for j in range(len(cpt_ctx[i])):
            count = sum([1 for word in cpt_ctx[i][j] if word in candidates])
            pos = len(cpt_ctx[i]) - j
            if pos not in distance_dict:
                distance_dict[pos] = [count, len(cpt_ctx[i][j])]
            else:
                distance_dict[pos][0] += count
                distance_dict[pos][1] += len(cpt_ctx[i][j])
    return distance_dict


# the ratio of concepts in response that are related to the context, sorted by distance
def distance_rate_by_response(cpt_res, cpt_per_utt, cn):
    distance_dict = {}
    cpt_ctx = [line[:-1] for line in cpt_per_utt]
    for i in range(len(cpt_res)):
        for j in range(len(cpt_ctx[i])):
            candidates, _ = cn.expand_list(cpt_ctx[i][j])
            count = sum([1 for word in cpt_res[i] if word in candidates])
            pos = len(cpt_ctx[i]) - j
            if pos not in distance_dict:
                distance_dict[pos] = [count, len(cpt_res[i])]
            else:
                distance_dict[pos][0] += count
                distance_dict[pos][1] += len(cpt_res[i])
    return distance_dict


# the shortest distance of a response to be guided
def guide_distance(cpt_res, cpt_per_utt, cn):

    def guide(res, ctx, cn):
        for word in res:
            cnt = sum([1 for cpt in ctx if word in cn.cpt_dict[cpt]])
            if cnt > 0:
                return True
        return False

    distance_dict = {}
    for i in range(len(cpt_res)):
        for j in range(len(cpt_per_utt[i]) - 1):
            pos = len(cpt_per_utt[i]) - j - 2
            if guide(cpt_res[i], cpt_per_utt[i][pos], cn):
                if j + 1 in distance_dict:
                    distance_dict[j + 1] += 1
                else:
                    distance_dict[j + 1] = 1
                break
    return distance_dict


# decay detection
def decay_detection(cpt_res, cpt_per_utt, cn):
    distance_dict = {}

    def transfer(per_utt):
        new_concepts = copy.deepcopy(per_utt)
        for i in range(len(new_concepts)):
            for word in per_utt[i]:
                for j in reversed(range(i+1, len(per_utt))):
                    if sum([1 for cpt in per_utt[j] if word in cn.cpt_dict[cpt]]) or word in per_utt[j]:
                        new_concepts[i].remove(word)
                        if word not in new_concepts[j]:
                            new_concepts[j].append(word)
                        break
        return new_concepts
    for i in range(len(cpt_res)):
        new_ctx = transfer(cpt_per_utt[i][:-1])
        for j in range(len(new_ctx)):
            pos = len(new_ctx) - j
            for word in new_ctx[j]:
                if sum([1 for cpt in cpt_res[i] if cpt in cn.cpt_dict[word]]):
                    if pos not in distance_dict:
                        distance_dict[pos] = [1, 1]
                    else:
                        distance_dict[pos][0] += 1
                if pos in distance_dict:
                    distance_dict[pos][1] += 1
                else:
                    distance_dict[pos] = [0, 1]
    return distance_dict


def get_state(ut, later_dialog, cn):
    state = []
    for cpt in ut:
        for later_ut in later_dialog:
            res = sum([1 if cpt in cn.cpt_dict[later_cpt] else 0 for later_cpt in later_ut])
            if res > 0:
                state.append(cpt)
                break
    return state


def filter_relate(corpus_per_utt, cn):
    corpus_filtered = []
    corpus_filtered_linear = []
    for dialog in corpus_per_utt:
        state_dialog = []
        state_linear = []
        for i, utt in enumerate(dialog):
            if i >= len(dialog) - 2:
                state_utt = utt
            else:
                state_utt = get_state(utt, dialog[i+1:], cn)
            state_dialog.append(state_utt)
            state_linear.extend(state_utt)
        corpus_filtered.append(state_dialog)
        corpus_filtered_linear.append(state_linear)
    return corpus_filtered, corpus_filtered_linear


def add_dependency(state_corpus, dep_corpus):
    punctuation = [',', '.', '-', '?', '!', "'", '"', 'ROOT']
    result = []
    for i in range(len(state_corpus)):
        result_dialog = []
        for j in range(len(state_corpus[i])):
            dict_utterance = {}
            for word in state_corpus[i][j]:
                related_dep = [dep[0] for dep in dep_corpus[i][j] if dep[1] == word and dep[0] not in punctuation]
                dict_utterance[word] = related_dep
            result_dialog.append(dict_utterance)
        result.append(result_dialog)
    return result


def main():
    vocab = pickle.load(open("vocab", "rb"))
    path = '../../../ConceptNet/valid_tokenized'
    dep_path = '../../../ConceptNet/valid_dependency'
    cn = ConceptNet("../../../ConceptNet/concept_dict_simple.json")
    stopwords = [word.strip() for word in open('../../../ConceptNet/stopword.txt').readlines()]
    data = pickle.load(open(path, "rb"))
    dep = pickle.load(open(dep_path, 'rb'))[1:]
    cpt_per_utt, cpt_linear = get_concepts(data, cn, stopwords)
    cpt_ctx, cpt_res = split_cpt(cpt_per_utt, 1)
    print("Processing completed.")
    state_corpus, state_linear = filter_relate(cpt_per_utt, cn)
    dep_corpus = add_dependency(state_corpus, dep)
    pickle.dump(dep_corpus, open("../../../ConceptNet/dep_valid", "wb"))
    write_file('../../../ConceptNet/valid.tsv', data, state_corpus, cpt_res)

#main()
