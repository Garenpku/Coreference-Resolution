class SyntaxNode:
    def __init__(self, label, start, end=-1):
        self.label = label
        self.start = start
        self.end = end
        self.subtree = []


def my_filter(sent):
    raw_list = sent.split(' ')
    res = []
    for word in raw_list:
        if word:
            res.append(word)
    return res


def construct_data(data):
    i = 0
    raw_corpus = []
    while i < len(data):
        sentence = []
        while i < len(data) and data[i] != "":
            if data[i].startswith("#end") or data[i].startswith("#begin"):
                i += 1
                continue
            sentence.append(data[i])
            i += 1
        if sentence:
            raw_corpus.append(sentence)
        i += 1
    corpus_filtered = [[my_filter(word) for word in sent] for sent in raw_corpus]
    cur_doc = '0'
    corpus = []
    discourse = []
    for sent in corpus_filtered:
        if sent[0][1] != cur_doc:
            cur_doc = sent[0][1]
            corpus.append(discourse)
            discourse = [sent]
        else:
            discourse.append(sent)
    corpus.append(discourse)
    return corpus


def dict_add(d, key, value):
    if key in d:
        d[key].append(value)
    else:
        d[key] = [value]


def get_coreference(discourse_coref, discourse_word, discourse_pos):
    coref_dict = {}
    cd_exist = []
    for sent_index, sent in enumerate(discourse_coref):
        activated = []
        activated_word = {}
        activated_pos = {}
        for word_index, word in enumerate(sent):
            this_word = discourse_word[sent_index][word_index]
            postag = discourse_pos[sent_index][word_index]
            for ac_word in activated:
                activated_word[ac_word].append(this_word)
                activated_pos[ac_word].append(str(word_index))
            if postag == 'CD':
                for ac_word in activated:
                    if ac_word not in cd_exist:
                        cd_exist.append(ac_word)
            if word == '-':
                continue
            all_coref = word.split('|')
            for coref in all_coref:
                if coref[0] == '(':
                    if coref[-1] == ')':
                        dict_add(coref_dict, int(coref[1:-1]), this_word + '|' + str(sent_index) + '|' + str(word_index))
                    else:
                        activated.append(int(coref[1:]))
                        activated_word[int(coref[1:])] = [this_word]
                        activated_pos[int(coref[1:])] = [str(word_index)]
                elif coref[-1] == ')':
                    activated.remove(int(coref[:-1]))
                    dict_add(coref_dict, int(coref[:-1]), " ".join(activated_word[int(coref[:-1])]) + '|' + str(sent_index) + '|' + "-".join(activated_pos[int(coref[:-1])]))
                    activated_word[int(coref[:-1])] = []
                    activated_pos[int(coref[:-1])] = []
    return coref_dict, cd_exist


def process_syntax_tree(syntax_sentence):
    top_node = SyntaxNode("TOP_OF_SENTENCE", 0)
    subtree_stack = [top_node]
    for i, word in enumerate(syntax_sentence):
        left_bracket, right_bracket = word.split('*')
        if left_bracket:
            new_subtrees = left_bracket.split('(')[1:]
            for subtree in new_subtrees:
                subtree_stack.append(SyntaxNode(subtree, i))
                subtree_stack[-2].subtree.append(subtree_stack[-1])
        if right_bracket:
            num_close_subtree = len(right_bracket)
            for j in range(num_close_subtree):
                subtree_stack[-1].end = i
                subtree_stack = subtree_stack[:-1]
    assert len(subtree_stack) == 1
    return top_node


def extract_from_conll(path):
    file = open(path)
    data = [line.strip() for line in file.readlines()][1:]
    corpus = construct_data(data)
    coreference = [[[word[-1] for word in sent] for sent in discourse] for discourse in corpus]
    word = [[[word[3] for word in sent] for sent in discourse] for discourse in corpus]
    pos = [[[word[4] for word in sent] for sent in discourse] for discourse in corpus]
    syntax = [[[word[5] for word in sent] for sent in discourse] for discourse in corpus]
    coref_corpus = []
    cd_corpus = []
    for i in range(len(coreference)):
        coref_dict, cd_exist = get_coreference(coreference[i], word[i], pos[i])
        coref_corpus.append(coref_dict)
        cd_corpus.append(cd_exist)
    #syntax_document = []
    #for i in range(len(syntax[0])):
    #    syntax_document.append(process_syntax_tree(syntax[0][i]))
    #for i in range(len(coreference)):
        #for index in cd_corpus[i]:
            #print(coref_corpus[i][index])
    return word, coref_corpus, syntax
