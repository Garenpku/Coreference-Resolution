import json
import os
import re
import graphviz
from eval import *
from parse_eds import *
from extract_coref import process_syntax_tree


graphviz_color_list = ['blue', 'chartreuse', 'darkorange', 'deeppink', 'lightskyblue', 'pink', 'red', 'brown', 'yellow', 'indigo',
                       'teal', 'purple', 'darkgoldenrod3', 'cornsilk4', 'burlywood']


def sort_key(string):
    dir_num = re.findall('dir(.*?)-', string)[0]
    doc_num = re.findall('document(.*)', string)[0]
    return int(dir_num) * 1000 + int(doc_num)


def plot_one_sentence(dot, index_to_node, core, label=None, color=None):
    #dot_lines = ["subgraph cluster_subgrph" + subgraph_name + " {"]
    #dot_lines = ["{\nrankdir=LR"]
    dot_lines = ["START_OF_SUBGRAPH"]
    if not index_to_node:
        dot_lines.append('label=\"{}\"'.format(label + "SENTENCE NOT ALIGNED IN DEEPBANK."))
        dot_lines.append('}')
        return dot_lines
    if label:
        dot_lines.append('label=\"{}\"'.format(label))
    for node in index_to_node.values():
        if node.index.startswith('_'):
            #dot.node(node.index, node.index)
            continue
        elif node.index.startswith('x'):
            node.name = node.name.replace('"', '\\"')
            if node.index == core:
                dot_lines.append("{} [label=\"{}\" color={} shape=box]".format(node.index, node.name, color))
                #dot.node(node.index, node.name, shape='box', color='blue')
            else:
                dot_lines.append("{} [label=\"{}\" shape=box]".format(node.index, node.name))
                #dot.node(node.index, node.name, shape='box')
        else:
            node.name = node.name.replace('"', '\\"')
            dot_lines.append("{} [label=\"{}\"]".format(node.index, node.name))
            #dot.node(node.index, node.name)
    for node in index_to_node.values():
        if node.index.startswith('_'):
            #dot.edge(node.index, node.bv.index, 'BV')
            continue
        elif len(node.args):
            for i, arg in enumerate(node.args):
                #dot.edge(node.index, arg.index, 'arg' + str(i))
                dot_lines.append("{} -> {} [label={}]".format(node.index, arg.index, 'arg' + str(i)))
    dot_lines.append('}')
    return dot_lines
    #dot.view()


def plot_syntax_tree(top_node):
    top_node = top_node.subtree[0]
    top_node.index = 0
    stack = [top_node]
    dot_lines = ["digraph{"]
    dot_lines.append("{} [label=\"{}\" shape=box]".format(str(0), top_node.label))
    count = 1
    while stack:
        current_top = stack[-1]
        stack = stack[:-1]
        for child in current_top.subtree:
            child.index = count
            count += 1
            dot_lines.append("{} [label=\"{}\"]".format(str(count), child.label))
            dot_lines.append("{} -> {}".format(str(current_top.index), str(child.index)))
            stack.append(child)
    dot_lines.append('}')
    return dot_lines


def rename(index_to_node, rename_time=0):
    rename_label = 'r'
    for node in index_to_node.values():
        #parsed = re.findall('([xei_])(.*)')
        node.index = node.index + rename_time * rename_label


# 功能函数，计算关于代词的指标
def calculate_figures(all_list):
    file = open("pronoun.txt", "w")
    count_all = 0
    count_same_sentence = 0
    count_last_sentence = 0
    count_first_appear = 0
    for cnt, document in enumerate(all_list):
        #if cnt > 10:
        #    break
        file.write(document + '\n\n')
        with open('../db-on-align/' + document) as f:
            if not len(f.read()):
                flag = 0
            else:
                flag = 1
        if not flag:
            continue
        data = json.load(open('../db-on-align/' + document))
        all_coreference = data['coreference']
        all_eds = [parse_eds(line['eds'][1:-1]) for line in data['discourse']]
        syntax_raw = [line['syntax'] for line in data['discourse']]
        all_syntax = []
        for line in syntax_raw:
            all_syntax.append(process_syntax_tree(line))
        res = plot_syntax_tree(all_syntax[0])
        source = graphviz.Source("\n".join(res))
        source.render('syntax')

        for num, relation in enumerate(all_coreference):
            for cnt, mention in enumerate(relation):
                index = mention.split('|')[-1]
                sentence_id = int(mention.split('|')[1])
                if all_eds[sentence_id][index].name == 'pron':
                    count_all += 1
                    if cnt > 0 and int(relation[cnt-1].split('|')[1]) == sentence_id:
                        count_same_sentence += 1
                    else:
                        if cnt == 0:
                            count_first_appear += 1
                            file.write("Pronoun first mentioned.\n")
                            file.write("Current mention: " + mention.split('|')[0] + '\n')
                            file.write(str(sentence_id) + 'th sentence: ' + data['discourse'][sentence_id]['sentence'] + '\n\n')
                        else:
                            if int(relation[cnt-1].split('|')[1]) == sentence_id - 1:
                                count_last_sentence += 1
                            file.write("Pronoun in different sentence.\n")
                            file.write("Current mention: " + mention.split('|')[0] + '\n')
                            file.write(str(sentence_id) + 'th sentence: ' + data['discourse'][sentence_id]['sentence'] + '\n')
                            file.write("Last mention: " + relation[cnt-1].split('|')[0] + '\n')
                            file.write(relation[cnt-1].split('|')[1] + 'th sentence: ' + data['discourse'][int(relation[cnt-1].split('|')[1])]['sentence'] + '\n\n')
    print("All pronouns:", count_all)
    print("Pronouns with precedent appearing in the same sentence:", count_same_sentence)
    print("Pronouns with precedent in the last sentence:", count_last_sentence)
    print("Pronouns with no precedent:", count_first_appear)
    file.close()
    return count_all


# 功能函数，计算专有名词相关性质
def cal_proper_name(all_list):
    def compound_connect_check(compounds):
        nodes = []
        for node in compounds:
            nodes.append(node.args[0].index)
            nodes.append(node.args[1].index)
        nodes = list(set(nodes))
        nodes = [[item] for item in nodes]
        arg2_node = []
        for node in compounds:
            arg1 = node.args[0].index
            arg2 = node.args[1].index
            if arg1 == arg2:
                continue
            if node.index.startswith('i'):
                arg2_node.append(arg1)
            else:
                arg2_node.append(arg2)
            for i, index in enumerate(nodes):
                if arg1 in index:
                    pos1 = i
                    chunk1 = index
                if arg2 in index:
                    pos2 = i
                    chunk2 = index
            if pos1 == pos2:
                continue
            nodes.append(nodes[pos1] + nodes[pos2])
            nodes.remove(chunk1)
            nodes.remove(chunk2)
        head_node = [[index for index in chunk if index not in arg2_node] for chunk in nodes]
        return nodes, head_node

    def get_full_proper_name(index_to_node, proper_node):
        compounds = [node for node in index_to_node.values() if (isinstance(node, E) or isinstance(node, I)) and node.name.startswith('compound') and len(node.args) == 2]
        nodes, head_node = compound_connect_check(compounds)

        for i, chunk in enumerate(nodes):
            if proper_node.index in chunk:
                #return "|".join([index_to_node[idx].name for idx in nodes[i]])
                if proper_node.index != head_node[i][0]:
                    #print(proper_node.index)
                    #print(head_node)
                    return None
                else:
                    return "|".join([index_to_node[idx].name for idx in nodes[i]])
        return proper_node.name

    def check_name_arg2(index_to_node, proper_node, name_list):
        for pred in index_to_node.values():
            for name in name_list:
                if isinstance(pred, E) and pred.name.startswith(name) and len(pred.args) == 2 and pred.args[1] == proper_node:
                    return True
        return False

    def proper_name_resolution(system_cluster, reverse_system_cluster, all_eds):
        for i, eds in enumerate(all_eds):
            for node in eds.values():
                if node.index.startswith('x') and node.name.startswith('named'):
                    #full_name = get_full_proper_name(eds, node)
                    full_name = node.name
                    if not full_name:
                        """
                        print(node.index, node.name)
                        for l in data['discourse'][i]['eds']:
                            print(l, end='')
                        input()
                        """
                        continue
                    if check_name_arg2(eds, node, ["appos", "compound"]):
                        """
                        if not str(i) + '|' + node.index in pos_to_entity:
                            continue
                        with open("proper_name", "a+") as file:
                            file.write(node.name + ' ' + node.index + '\n')
                            file.write(data['discourse'][i]['sentence'] + '\n')
                            file.write(str(all_coreference[pos_to_entity[str(i) + '|' + node.index]]) + '\n')
                            for l in data['discourse'][i]['eds']:
                                file.write(l)
                        """
                        continue
                    key_name = str(i) + '|' + node.index
                    flag = 0
                    for key in system_cluster.keys():
                        if set(key.split('|')) == set(full_name.split('|')):
                            system_cluster[key].append(key_name)
                            reverse_system_cluster[key_name] = key
                            flag = 1
                    if not flag:
                        system_cluster[full_name] = [key_name]
                        reverse_system_cluster[key_name] = full_name

    def noun_phrase_resolution(system_cluster, reverse_system_cluster, all_eds):
        for i, eds in enumerate(all_eds):
            for node in eds.values():
                if node.index.startswith('x') and re.findall("_(.*?)_n", node.name):
                    full_name = get_full_proper_name(eds, node)
                    #full_name = node.name
                    if not full_name:
                        continue
                    key_name = str(i) + '|' + node.index
                    flag = 0
                    for key in system_cluster.keys():
                        if set(key.split('|')) == set(full_name.split('|')):
                            system_cluster[key].append(key_name)
                            reverse_system_cluster[key_name] = key
                            flag = 1
                    if not flag:
                        system_cluster[full_name] = [key_name]
                        reverse_system_cluster[key_name] = full_name

    recall = 0
    precision = 0
    F = 0
    count = 0
    for cnt, document in enumerate(all_list):
        #if cnt > 0:
        #    break
        with open('../db-on-align/' + document) as f:
            if not len(f.read()):
                flag = 0
            else:
                flag = 1
        if not flag:
            continue
        count += 1
        data = json.load(open('../db-on-align/' + document))
        #file.write("\nNew Document:" + document + '\n')
        all_coreference = data['coreference']
        all_eds = [parse_eds(line['eds'][1:-1]) for line in data['discourse']]
        mention_to_entity = {}
        pos_to_entity = {}
        entity_to_pos = {}
        all_mentions = []
        for i, relation in enumerate(all_coreference):
            processed = []
            entity_to_pos[i] = []
            for mention in relation:
                node_name = all_eds[int(mention.split('|')[1])][mention.split('|')[-1]].name
                #if not node_name.startswith('named'):
                if not re.findall("_(.*?)_n", node_name):
                    continue
                key_name = mention.split('|')[1] + '|' + mention.split('|')[-1]
                pos_to_entity[key_name] = i
                entity_to_pos[i].append(key_name)
                if node_name in mention_to_entity and mention_to_entity[node_name] != i and node_name not in processed:
                    """
                    file.write(node_name + '\n')
                    file.write(str(mention_to_entity[node_name]) + str(all_coreference[mention_to_entity[node_name]]) + '\n')
                    file.write(str(i) + str(relation) + '\n')
                    """
                    processed.append(node_name)
                else:
                    mention_to_entity[node_name] = i
                all_mentions.append(all_eds[int(mention.split('|')[1])][mention.split('|')[-1]].name)
        all_mentions = list(set(all_mentions))

        system_cluster = {}
        reverse_system_cluster = {}
        #proper_name_resolution(system_cluster, reverse_system_cluster, all_eds)
        noun_phrase_resolution(system_cluster, reverse_system_cluster, all_eds)

        """
        print(pos_to_entity)
        print(entity_to_pos)
        print(system_cluster)
        print(reverse_system_cluster)
        """
        if not pos_to_entity:
            count -= 1
            continue

        score = B3(entity_to_pos, pos_to_entity, system_cluster, reverse_system_cluster)

        """
        # ALL-IN-ONE
        one_in_all = {1:list(pos_to_entity.keys())}
        all_in_one = {}
        for pos in one_in_all[1]:
            all_in_one[pos] = 1
        score = B3(entity_to_pos, pos_to_entity, one_in_all, all_in_one)
        """

        recall += score[0]
        precision += score[1]
        F += score[2]
    recall /= count
    precision /= count
    F /= count
    print("Precision:", precision)
    print("Recall:", recall)
    print("F:", F)


# 功能函数，将eds绘制成图
def plot_corpus(all_list):
    duplicated_core = 0
    for cnt, document in enumerate(all_list):
        color_to_surface = {}
        if cnt > 10:
            break
        print(document)
        data = json.load(open("../db-on-align/" + document))
        all_coreference = data['coreference']
        #dot_source = ['digraph{\nnetwork=true']
        dot_source = []
        processed_sentence = []
        for num, relation in enumerate(all_coreference):
            if num >= len(graphviz_color_list):
                break
            dot = graphviz.Digraph()
            for i in range(len(relation)):
                print(relation[i])
                sentence_index = int(relation[i].split('|')[1])
                core = relation[i].split('|')[3]
                surface_form = relation[i].split('|')[0]
                color = graphviz_color_list[num]
                if color in color_to_surface:
                    color_to_surface[color].append(surface_form)
                else:
                    color_to_surface[color] = [surface_form]
                eds = parse_eds(data['discourse'][sentence_index]['eds'][1:-1])
                rename(eds, len(processed_sentence))
                core = core + len(processed_sentence) * 'r'
                if sentence_index in processed_sentence:
                    index = processed_sentence.index(sentence_index)
                    core = core[:core.index("r")] + index * 'r'
                    print("Sentence already exists. The core is:", core)
                    for j, line in enumerate(dot_source):
                        #if line.startswith(core) and not '->' in line:
                        if line.split(' ')[0] == core and '->' not in line:
                            label = re.findall('label="(.*)"', line)[0]
                            #dot_source[j] = dot_source[j].replace(label, label + '|' + surface_form)
                            dot_source[j] = dot_source[j][:-1] + ' color=' + graphviz_color_list[num] + ']'
                    continue
                print("New sentence. Start a new subgraph.")

                dot_lines = plot_one_sentence(dot, eds, core, label="Sentence " + str(sentence_index) + ": " + data['discourse'][sentence_index]['sentence'],
                                              color=graphviz_color_list[num])
                dot_source.extend(dot_lines)
                processed_sentence.append(sentence_index)
            print("END OF RELATION\n")

        # 检查是否一个结点带两种颜色
        for line in dot_source:
            if len(re.findall("color=(.*?)[ \]]", line)) > 1:
                print(line)
                duplicated_core += 1

        color_to_node = {}
        for line in dot_source:
            color = re.findall('color=(.*?)[ \]]', line)
            if color:
                name = re.findall('label=(.*?) ', line)[0]
                if color[0] in color_to_node:
                    if name not in color_to_node[color[0]]:
                        color_to_node[color[0]].append(name)
                else:
                    color_to_node[color[0]] = [name]
        print(color_to_node)
        color_source = ['digraph{']
        label_assigner = 0
        for item in color_to_node.items():
            for name in item[1]:
                dot_line = "c" + str(label_assigner) + " [label={} color={}]".format(name, item[0])
                label_assigner += 1
                color_source.append(dot_line)
        label_assigner = 0
        for item in color_to_node.items():
            for k in range(len(item[1])):
                #dot_line = re.findall('"(.*)"', item[1][k])[0] + ' -> ' + re.findall('"(.*)"', item[1][k+1])[0]
                if k == len(item[1]) - 1:
                    label_assigner += 1
                    continue
                dot_line = "c" + str(label_assigner) + ' -> c' + str(label_assigner + 1) + ' [label="coref"]'
                label_assigner += 1
                color_source.append(dot_line)
        color_source.append('}')
        source = graphviz.Source("\n".join(color_source))
        source.render('../aligned_graph/' + document + '/color')

        color_source = ['digraph{']
        label_assigner = 0
        for item in color_to_surface.items():
            for name in item[1]:
                dot_line = "c" + str(label_assigner) + " [label=\"{}\" color={}]".format(name.replace(' ', '_'), item[0])
                label_assigner += 1
                color_source.append(dot_line)
        label_assigner = 0
        for item in color_to_surface.items():
            for k in range(len(item[1])):
                #dot_line = re.findall('"(.*)"', item[1][k])[0] + ' -> ' + re.findall('"(.*)"', item[1][k+1])[0]
                if k == len(item[1]) - 1:
                    label_assigner += 1
                    continue
                dot_line = "c" + str(label_assigner) + ' -> c' + str(label_assigner + 1) + ' [label="coref"]'
                label_assigner += 1
                color_source.append(dot_line)
        color_source.append('}')
        source = graphviz.Source("\n".join(color_source))
        source.render('../aligned_graph/' + document + '/color_surface')

        #subgraphs = re.findall('START_OF_SUBGRAPH\n((?:.|\n)*?)START_OF_SUBGRAPH', "\n".join(dot_source) + "START_OF_SUBGRAPH")
        subgraphs = "\n".join(dot_source).split('START_OF_SUBGRAPH\n')[1:]
        sort_dict = {}
        for k, line in enumerate(subgraphs):
            index = int(re.findall("Sentence (.*?):", line)[0])
            sort_dict[index] = k
        #subgraphs.sort(key=lambda x: int(re.findall("Sentence (.*?):", x)[0]), reverse=False)
        print(len(subgraphs))

        for k in range(len(data['discourse'])):
            if k not in sort_dict:
                eds = parse_eds(data['discourse'][k]['eds'][1:-1])
                sentence = data['discourse'][k]['sentence']
                dot_lines = plot_one_sentence(None, eds, "", label="Sentence " + str(k) + ": " + sentence)[1:]
                subgraph = "\n".join(dot_lines)
                source = graphviz.Source("digraph{\nlabelloc=t\nfontsize=20\n" + subgraph)
                if k < 10:
                    source.render('../aligned_graph/' + document + '/sentence-0' + str(k))
                else:
                    source.render('../aligned_graph/' + document + '/sentence-' + str(k))
                continue
            subgraph = subgraphs[sort_dict[k]]
            if not os.path.exists("../aligned_graph/" + document):
                os.mkdir("../aligned_graph/" + document)
            source = graphviz.Source("digraph{\nlabelloc=t\nfontsize=20\n" + subgraph)
            if k < 10:
                source.render('../aligned_graph/' + document + '/sentence-0' + str(k))
            else:
                source.render('../aligned_graph/' + document + '/sentence-' + str(k))
        os.chdir("../aligned_graph/" + document)
        os.system("pdftk color.pdf color_surface.pdf sentence-*.pdf cat output " + document + "-combined.pdf")
        os.system("cp " + document + "-combined.pdf ../all_result")
        os.chdir("../../deepbank")
    print(duplicated_core)
        #dot.view()


all_list = sorted([d for d in os.listdir("../db-on-align") if d.startswith('dir')], key=lambda x: sort_key(x))
#plot_corpus(all_list)
print(calculate_figures(all_list))
#cal_proper_name(all_list)
