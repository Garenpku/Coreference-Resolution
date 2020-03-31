import os
import re
import json
from extract_coref import *
from delphin.mrs.eds import loads_one
from parse_eds import *
import traceback
ontonote = '../OntoNotes-5.0-NER-BIO/conll-formatted-ontonotes-5.0/data/wsj'

debug = 0
not_matched_chunk = 0


def read_one_file(path):
    ls = open(path).readlines()
    return [line.strip() for line in ls]


def get_sentence_from_list(data_list):
    for i, line in enumerate(data_list):
        if line == '<':
            j = i + 1
            words = []
            indexes = []
            while data_list[j] != '>':
                word = re.findall('"(.*?)"', data_list[j])[0]
                words.append(word)
                index = re.findall('<(.*?)>', data_list[j])[0]
                indexes.append(index)
                j += 1
            return words, indexes
    #return data_list[5][data_list[5].index('`')+1:-1]


def get_corpus_one_dir(direction, cname='deepbank'):
    if cname == 'deepbank':
        files = os.listdir(direction)
        files = list(sorted([file for file in files if len(file) == 8]))
        words = [get_sentence_from_list(read_one_file(direction + '/' + file))[0] for file in files]
        index = [get_sentence_from_list(read_one_file(direction + '/' + file))[1] for file in files]
        eds = [open(direction + '/' + file + '-eds').readlines()[2:] for file in files]
        res = (words, index, eds)
    elif cname == 'ontonote':
        files = os.listdir(direction)
        files = list(sorted([file for file in files if file.endswith('conll')]))
        tmp = [extract_from_conll(direction + '/' + file) for file in files]
        words = [file[0] for file in tmp]
        coref = [file[1] for file in tmp]
        syntax = [file[2] for file in tmp]
        res = (words, coref, syntax)
    return res


def get_n_dir(start, n, cname='deepbank'):
    wsj_list = [name for name in sorted(os.listdir('.')) if name.startswith('wsj')][start:start+n]
    if cname == 'deepbank':
        db = []
        index = []
        eds = []
        for name in wsj_list:
            db_tmp, index_tmp, eds_tmp = get_corpus_one_dir(name, cname)
            db.extend(db_tmp)
            index.extend(index_tmp)
            eds.extend(eds_tmp)
        return (db, index, eds)
    elif cname == 'ontonote':
        return 0


def flatten_words(word_direction):
    res = []
    for file in word_direction:
        for doc in file:
            for sent in doc:
                res.append(" ".join(sent))
    return res


def search(src, dst, threshold=0.9):
    def match_rate(sent1, sent2):
        return 2 * len(set(sent1).intersection(sent2)) / (len(set(sent1)) + len(set(sent2)))
        #return sum([1 for j in range(min(len(sent1), len(sent2))) if sent1[j] == sent2[j]]) / len(sent1)
    max_rate = 0
    index = -1
    for i, sent in enumerate(dst):
        rate = match_rate(src, sent)
        if rate > max_rate:
            index = i
            max_rate = rate
    if max_rate > threshold:
        return index
    else:
        return -1


def judge_document_cover(on_file, db):
    res = []
    for i, doc in enumerate(on_file):
        flag = 1
        for sent in doc:
            if sent not in db:
                flag = 0
        if flag:
            res.append(i)
    return res


# the first argument: a diction of coreference information
# the second argument: a list of eds lists, where missing sentences is padded by empty lists
def relate_eds_coref(coref_discourse, eds_discourse, db_discourse, index_discourse, syntax_discourse):

    def wash_name(name):
        # 将"-"两端连起来作为一个词
        name_list = name.split(' ')
        new_list = []
        i = 0
        while i < len(name_list):
            word = name_list[i]
            word = word.replace("'", "’")
            word = word.replace('``', '“')
            word = word.replace('’’', '”')
            word = word.replace('-LCB-', '{')
            word = word.replace('-RCB-', '}')
            if word == '-':
                new_list[-1] = new_list[-1] + '-' + name_list[i+1]
                i += 1
            elif word == '--':
                new_list.append('–')
            elif word == '-LRB-':
                new_list.append('(')
            elif word == '-RRB-':
                new_list.append(')')
            else:
                new_list.append(word)
            i += 1
        return " ".join(new_list)

    def match_name_sent(initial_name, sent, index_sent, span_index=None):
        sent_str = " ".join(sent)
        name = wash_name(initial_name)
        if not name:
            print("Initial name for void:", initial_name)
        #print("Washed: ", name)
        name_list = name.split(' ')
        accumulated_pos = 0
        result = []
        for i in range(len(sent)-len(name_list)+1):
            flag = 1
            for j in range(i, i + len(name_list)):
                if sent[j] != name_list[j-i]:
                    flag = 0
            if flag:
                start = int(index_sent[i].split(':')[0])
                end = int(index_sent[i+len(name_list)-1].split(':')[-1])
                #return start, end
                result.append([start, end, i])

        # 选择最近的span
        if len(result):
            span_start = int(span_index.split('-')[0])
            distance = 10000
            for k, span in enumerate(result):
                if abs(span[2] - span_start) < distance:
                    distance = abs(span_start - span[2])
                    record = k
            return result[record][:2]

        # e.g. Packwood & Packwood-Roth
        if len(name_list) == 1:
            for i in range(len(sent)):
                if name_list[0] in sent[i]:
                    start = int(index_sent[i].split(':')[0])
                    end = int(index_sent[i].split(':')[-1])
                    return start, end
        if 'U.S .' in sent_str and 'U.S.' in name:
            name = name.replace('U.S.', 'U.S .')
            name_list = name.split(' ')
            print(name_list)
            print(name in sent_str)
            for i in range(len(sent)-len(name_list) + 1):
                flag = 1
                for j in range(i, i + len(name_list)):
                    print(sent[j])
                    print(name_list[j-i])
                    print('')
                    if sent[j] != name_list[j-i]:
                        flag = 0
                if flag:
                    start = int(index_sent[i].split(':')[0])
                    end = int(index_sent[i+len(name_list)-1].split(':')[-1])
                    return start, end

        print("Washed:", name)
        print(sent_str)
        print("")
        return -1, -1

    def get_related_node(start, end, nodes, sent=None):
        res = []
        for item in nodes.items():
            item_start = item[1].lnk.data[0]
            item_end = item[1].lnk.data[1]
            if start - 2 <= item_start and end + 2 >= item_end and ((item_start <= end and item_end >= start) or (item_end >= start and item_start <= end)):
                res.append(item[0])
            #elif ((start <= item_start <= end) or (start <= item_end <= end)) and not item[0].startswith('e'):
            elif ((start <= item_start <= end) or (start <= item_end <= end)) and item[0].startswith('e'):
                res.append(item[0])
        return res

    basic_information = []
    for i in range(len(db_discourse)):
        basic_information.append({'sentence': " ".join(db_discourse[i]), 'eds': eds_discourse[i], 'syntax': syntax_discourse[i]})
    global_dict['discourse'] = basic_information

    ambiguous = 0
    count_all = 0
    global not_matched_chunk
    coref_information_discourse = []
    for value in coref_discourse.values():
        coref_information = []
        for item in value:
            name = item.split('|')[0]
            sent_index = int(item.split('|')[1])
            span_index = item.split('|')[2]
            sent = db_discourse[sent_index]
            if not sent:
                continue
            start, end = match_name_sent(name, sent, index_discourse[sent_index], span_index)
            if start == -1 and end == -1:
                not_matched_chunk += 1
            eds_literal = " ".join(eds_discourse[sent_index])
            eds_literal = re.sub("\{.*\}", "", eds_literal)
            try:
                eds = loads_one(eds_literal)
            except:
                print(eds_literal)
                raise AssertionError('Error')
            related_nodes = get_related_node(start, end, eds._nodes, " ".join(sent))
            subgraph = []
            for node in related_nodes:
                for line in eds_discourse[sent_index]:
                    if line.startswith(' ' + node):
                        subgraph.append(line)
                        break
            try:
                index_to_node = parse_eds(subgraph)
                context = parse_eds(eds_discourse[sent_index][1:-1])
            except:
                for line in eds_discourse[sent_index][1:-1]:
                    print(line, end='')
                traceback.print_exc()
                exit(0)
            try:
                """
                if len([node for node in index_to_node.values() if node.index.startswith('i') and node.name == 'compound']):
                    print(name, '***', sent_index, ' ', " ".join(db_discourse[sent_index]), '\nSubgraph:')
                    for line in subgraph:
                        print(line, end='')
                    input('Check This.')
                """
                cores, reference_count = match_pattern(index_to_node, context)
            except:
                print("Error occurred when extracting cores!")
                print(name, '***', sent_index, ' ', " ".join(db_discourse[sent_index]), '\nSubgraph:')
                for line in subgraph:
                    print(line, end='')
                traceback.print_exc()
                exit(0)
            count_all += 1
            if len(cores) > 1 or len(cores) == 0:
                check = [1 for cnt in reference_count if cnt > 0]
                if len(check) == 1:
                    idx = [int(cnt > 0) * 1 for cnt in reference_count].index(1)
                    coref_information.append(item + '|' + cores[idx].index)
                    continue
                if len(cores) > 1:
                    ambiguous += 1
                else:
                    not_matched_chunk += 1
                if debug:
                    print(start, end)
                    if db_discourse:
                        print(name, '***', sent_index, ' ', " ".join(db_discourse[sent_index]), '\nSubgraph:')
                    else:
                        print(name, '***', sent_index, '\nSubgraph:')
                    for line in subgraph:
                        print(line, end='')
                    print('*******')
                    for line in eds_discourse[sent_index][1:-1]:
                        print(line, end='')
                    for j, node in enumerate(cores):
                        print(node.index, reference_count[j])
                    #for line in eds_discourse[sent_index]:
                    #    print(line, end='')
                    #print(eds_discourse[sent_index])
                    if str(input("Error occured. Check this. Continue? y/n\n")) != 'y':
                        exit(0)
                    print('=============')
            else:
                coref_information.append(item + '|' + cores[0].index)
        #if len(coref_information) > 1:
        #    for item in coref_information:
                #file.write(item + '\n')
            #file.write('\n')
        coref_information_discourse.append(coref_information)
    global_dict['coreference'] = coref_information_discourse
    json.dump(global_dict, file)
    return ambiguous, count_all


def pad_deepbank(db, on_discourse, ratio=0.7):
    res = []
    for i in range(len(on_discourse)):
        res.append(search(on_discourse[i], db, ratio))
    return res


def check_align(pad_index):
    valid_index = [index for index in pad_index if index != -1]
    if len(valid_index) / len(pad_index) < 0.5:
        return 0
    for i in range(len(valid_index) - 1):
        if valid_index[i+1] - valid_index[i] < 0:
            return 1
    return 2


def print_information(db, word_index, eds, on_discourse, coref_discourse, syntax_discourse):
    pad_index = pad_deepbank(db, on_discourse, 0.7)
    print(pad_index)
    result = check_align(pad_index)
    if not check_align(pad_index):
        print('Error')
        return 0
    # 还是有不连续的情况
    elif result == 1:
        print('Retry')
        pad_index = pad_deepbank(db, on_discourse, 0.5)
        result = check_align(pad_index)
        if result != 2:
            print('Error')
            #exit(0)
    eds_discourse = []
    db_discourse = []
    index_discourse = []
    for index in pad_index:
        if index == -1:
            eds_discourse.append([])
            db_discourse.append([])
            index_discourse.append([])
        else:
            eds_discourse.append(eds[index])
            db_discourse.append(db[index])
            index_discourse.append(word_index[index])
    ambiguous, count_all = relate_eds_coref(coref_discourse, eds_discourse, db_discourse, index_discourse, syntax_discourse)
    return ambiguous, count_all

# latest history: 13,46

"""
count_files = 0
count_documents = 0
for num, direction in enumerate(sorted(onto_dirs)):
    if direction.startswith('.'):
        continue
    on, coref = get_corpus_one_dir(ontonote + '/' + direction, 'ontonote')
    for i in range(len(coref)):
        valid = [doc for doc in coref[i] if len(doc)]
        assert len(valid) < 2
        for doc in coref[i]:
            if doc:
                count_documents += 1
        if coref[i][0]:
            count_files += 1
print(count_documents)
print(count_files)
"""
amb = 0
count_all = 0
onto_dirs = os.listdir(ontonote)
start = 0
cnt = 0


for num, direction in enumerate(sorted(onto_dirs)):
    if cnt > 0:
        cnt -= 1
        continue
    if direction.startswith('.'):
        continue
    on, coref, syntax = get_corpus_one_dir(ontonote + '/' + direction, 'ontonote')
    checkpoint = 0
    for i in range(len(coref)):
        if len(coref[i][0]):
            print("OntoNote Document", str(i), "Deepbank Direction", str(start))
            file = open('../db-on-align/dir{}-document{}'.format(str(num), str(i)), 'w')
            global_dict = {}
            while 1:
                db, index, eds = get_n_dir(start=start, n=2)
                res = print_information(db, index, eds, on[i][0], coref[i][0], syntax[i][0])
                if res:
                    amb += res[0]
                    count_all += res[1]
                    print("Ambiguous:", amb, "All:", count_all, "Ratio:", amb / count_all)
                    print("Not matched chunks:", not_matched_chunk)
                    checkpoint = 0
                    break
                else:
                    if checkpoint == 0:
                        checkpoint = start
                    start += 1
                    if start - checkpoint > 20:
                        print("OntoNote Document", str(i), "not found.")
                        start = checkpoint
                        print("Back to deepbank directory", str(start))
                        checkpoint = 0
                        break
            file.close()
    print("One direction in OntoNote is finished.", "This is the", str(num), "th direction.")
    print("Current deepbank index:", start)

