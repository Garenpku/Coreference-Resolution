import re

eds = """e11:appos<21:108>{e SF prop, TENSE untensed, MOOD indicative, PROG -, PERF -}[ARG1 x9, ARG2 x10]
 _2:proper_q<21:48>[BV x9]
 e17:compound<21:42>{e SF prop, TENSE untensed, MOOD indicative, PROG -, PERF -}[ARG1 x9, ARG2 x16]
 _3:proper_q<21:30>[BV x16]
 x16:named<21:30>("Darkhorse"){x PERS 3, NUM sg, IND +}[]
 x9:named<31:42>("Productions"){x PERS 3, NUM sg, IND +}[]
 x23:_inc_n_1<43:48>{x IND +}[]
 _4:udef_q<43:48>[BV x23]
 i27:compound<43:48>{i}[ARG1 x9, ARG2 x23]
 _5:_a_q<49:50>[BV x10]
 e33:compound<51:72>{e SF prop, TENSE untensed, MOOD indicative, PROG -, PERF -}[ARG1 x10, ARG2 x32]
 _6:udef_q<51:64>[BV x32]
 e39:compound<51:64>{e SF prop, TENSE untensed, MOOD indicative, PROG -, PERF -}[ARG1 x32, ARG2 x38]
 _7:proper_q<51:53>[BV x38]
 x38:named<51:53>("TV"){x IND +}[]
 x32:_production_n_of<54:64>{x}[]
 x10:_company_n_of<65:72>{x PERS 3, NUM sg}[]
 e46:_in_p_state<73:75>{e SF prop, TENSE untensed, MOOD indicative, PROG -, PERF -}[ARG1 e47, ARG2 x10]
 _8:proper_q<82:93>[BV x50]
 e54:compound<82:93>{e SF prop, TENSE untensed, MOOD indicative, PROG -, PERF -}[ARG1 x50, ARG2 x53]
 _9:udef_q<82:85>[BV x53]
 x53:_mister_n_1<82:85>{x PERS 3, NUM sg, IND +}[]
 x50:named<86:93>("Trudeau"){x PERS 3, NUM sg, IND +}[]
 e47:_be_v_id<94:96>{e SF prop, TENSE pres, MOOD indicative, PROG -, PERF -}[ARG1 x50, ARG2 x59]
 _10:_a_q<97:98>[BV x59]
 x59:_co-owner/nn_u_unknown<99:108>{x PERS 3, NUM sg}[]"""

context = """e4:_but_c<0:3>{e SF prop, TENSE untensed, MOOD indicative, PROG -, PERF -}[R-HNDL e9, R-INDEX e4]"""


class Q:
    def __init__(self, index, bv):
        self.index = index
        self.bv = bv


class X:
    def __init__(self, index, args, name):
        self.index = index
        self.args = args
        self.name = name


class E:
    def __init__(self, index, args, name):
        self.index = index
        self.args = args
        self.name = name


class I:
    def __init__(self, index, args, name):
        self.index = index
        self.args = args
        self.name = name


def parse_eds(eds):
    index_to_node = {}
    for line in eds:
        # quantifier
        while line.startswith(' '):
            line = line.strip()
        index = line[:line.index(':')]
        if line.startswith('_'):
            bv = re.findall('\[BV (.*?)\]', line)
            # 纠错机制
            if not bv:
                additional_info = re.findall('\{(.*?)\}', line)
                if len(additional_info):
                    if additional_info[0][0] == 'e':
                        line = 'e' + line[1:]
                    elif additional_info[0][0] == 'x':
                        line = 'x' + line[1:]
                else:
                    continue
            else:
                new_node = Q(index, bv[0])
        if line.startswith('x'):
            info = re.findall('\[(.*?\])', line)[0]
            pred_name = re.findall(':(.*?)<', line)[0]
            if pred_name == "named" or pred_name == "named_n":
                proper_name = re.findall('\(.*?\)', line)[0]
                pred_name = pred_name + proper_name
            if pred_name == "implicit_conj" or pred_name == '_and_c' or pred_name == '_or_c':
                args = re.findall('[LR]-INDEX (.*?)[,\]]', info)
            else:
                args = re.findall('ARG[1-9] (.*?)[,\]]', info)
            new_node = X(index, args, pred_name)
        if line.startswith('e'):
            info = re.findall('\[(.*?\])', line)[0]
            #args = re.findall('ARG[1-9] (.*?)[,\]]', info)
            args = [group[-1] for group in re.findall('(ARG[1-9]|[LR]-(INDEX|HNDL)) (.*?)[,\]]', info)]
            pred_name = re.findall(':(.*?)<', line)[0]
            new_node = E(index, args, pred_name)
        if line.startswith('i'):
            info = re.findall('\[(.*?\])', line)[0]
            args = re.findall('ARG[1-9] (.*?)[,\]]', info)
            pred_name = re.findall(':(.*?)<', line)[0]
            new_node = I(index, args, pred_name)
        index_to_node[index] = new_node
    for (index, node) in index_to_node.items():
        if isinstance(node, Q):
            if node.bv in index_to_node:
                node.bv = index_to_node[node.bv]
        else:
            for i, arg in enumerate(node.args):
                if arg in index_to_node:
                    node.args[i] = index_to_node[arg]
    return index_to_node


def match_pattern(index_to_node, context=None):
    def compound_connect_check(predicates):
        compounds = [node for node in predicates if node.name == 'compound']

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

    def expand_graph(predicates, start_entity, end_entity):
        working_set = [start_entity]
        pred_to_delete = []
        processed = []
        while len(working_set):
            entity = working_set[0]
            if isinstance(entity, str):
                working_set.remove(entity)
                continue
            if len(entity.args) != 0:
                working_set.extend([node for node in entity.args if node not in processed and node not in working_set])
                if isinstance(entity, E):
                    pred_to_delete.append(entity)
            for pred in predicates:
                if entity in pred.args and end_entity not in pred.args:
                    working_set.extend([node for node in pred.args if node != start_entity and node not in processed and node not in working_set])
                    pred_to_delete.append(pred)
            processed.append(entity)
            working_set.remove(entity)
        return processed, pred_to_delete

    def judge_outscope(node):
        for arg in node.args:
            if isinstance(arg, str):
                return True
        return False

    subgraph_index = [item[1].index for item in index_to_node.items()]
    context = list(context.values())
    context = [node for node in context if node.index not in subgraph_index]

    predicates = []
    entities = []

    # only event coreference
    initial_predicates = [item[1] for item in index_to_node.items() if item[0].startswith('e')]
    initial_entities = [item[1] for item in index_to_node.items() if item[0].startswith('x') or item[0].startswith('i')]
    function_list = ['parg_d', 'subord']
    if len(initial_entities) == 0:
        if len(initial_predicates) == 1:
            return initial_predicates, []
        initial_predicates = [pred for pred in initial_predicates if pred.name not in function_list]
        if len(initial_predicates) == 1:
            return initial_predicates, []

    for (index, node) in index_to_node.items():
        if index.startswith('e') and len(node.args) > 1:
            predicates.append(node)
            for arg in node.args:
                if isinstance(arg, E) and len(arg.args) <= 1 and arg not in predicates:
                    predicates.append(arg)
        elif index.startswith('x'):
            entities.append(node)
        elif index.startswith('i') and node.name == 'compound':
            predicates.append(node)
    context_predicate = [pred for pred in predicates if judge_outscope(pred)]
    predicates = [pred for pred in predicates if not judge_outscope(pred)]
    context = context + context_predicate

    """
    check connected subgraphs under compounding
    """
    connected_graph, head_node = compound_connect_check(predicates)
    for i in range(len(connected_graph)):
        if head_node[i][0] not in index_to_node:
            continue
        for node in connected_graph[i]:
            if node != head_node[i][0] and node in index_to_node:
                ent_to_del, pred_to_del = expand_graph(predicates, index_to_node[node], index_to_node[head_node[i][0]])
                entities = [node for node in entities if node not in ent_to_del]
                predicates = [pred for pred in predicates if pred not in pred_to_del]

    #entities = [node for node in entities if
    #            node.index not in connected_graph_flatten or node.index in head_node_flatten]
    predicates = [pred for pred in predicates if pred.name != "compound"]

    """
    check poss
    """
    poss_arg2 = []
    poss_predicates = []
    for pred in predicates:
        if pred.name == 'poss':
            arg1 = pred.args[0].index
            arg2 = pred.args[1].index
            if arg1 in index_to_node:
                poss_arg2.append(arg2)
                poss_predicates.append(pred.index)
            else:
                assert len(entities) == 1
    entities = [node for node in entities if node.index not in poss_arg2]
    predicates = [pred for pred in predicates if pred.index not in poss_predicates]

    """
    check appos
    """
    flag = 0
    for pred in predicates:
        if pred.name == 'appos':
            appos_subgraph, pred_to_delete = expand_graph(predicates, pred.args[1], pred.args[0])
            pred_to_delete.append(pred)
            flag = 1
            entities = [node for node in entities if node not in appos_subgraph]
            predicates = [pred for pred in predicates if pred not in pred_to_delete]

    """
    check preposition
    """
    preposition = ['_' + element + '_p' for element in ['of', 'with', 'by', 'for', 'from', 'against', 'in', 'through', 'since', 'on',
                                                        'over']]
    for pred in predicates:
        if pred.name in preposition:
            prep_subgraph, pred_to_delete = expand_graph(predicates, pred.args[1], pred.args[0])
            pred_to_delete.append(pred)
            entities = [node for node in entities if node not in prep_subgraph]
            predicates = [pred for pred in predicates if pred not in pred_to_delete]

    """
    check and
    """
    node_to_del = []
    pred_to_del = []
    for entity in entities:
        if entity.name == '_and_c' or entity.name == 'implicit_conj':
            flag = 0
            if isinstance(entity.args[0], X):
                expanded_node, expanded_pred = expand_graph(predicates, entity.args[0], entity)
                node_to_del.extend(expanded_node)
                pred_to_del.extend(expanded_pred)
                flag = 1
            if len(entity.args) > 1 and isinstance(entity.args[1], X):
                expanded_node, expanded_pred = expand_graph(predicates, entity.args[1], entity)
                node_to_del.extend(expanded_node)
                pred_to_del.extend(expanded_pred)
                flag = 1
            if not flag:
                entities.remove(entity)

    entities = [node for node in entities if node not in node_to_del]
    predicates = [pred for pred in predicates if pred not in pred_to_del]

    """
    check to [e.g. legislation to protect intellectual property]
    """
    for entity in entities:
        if len(entity.args) == 1:
            if isinstance(entity.args[0], E):
                ent_to_del, pred_to_del = expand_graph(predicates, entity.args[0], entity)
                entities = [node for node in entities if node not in ent_to_del]
                predicates = [pred for pred in predicates if pred not in pred_to_del]

    if not context:
        return entities, []

    """
    check nominalization 这个是不对的哈
    """
    #entities = [node for node in entities if node.name != 'nominalization']

    """
    check reference time from context
    """
    reference_count = []
    for entity in entities:
        count = 0
        for node in context:
            if isinstance(node, X) or isinstance(node, E) or isinstance(node, I):
                for arg in node.args:
                    if arg.index == entity.index:
                        count += 1
        reference_count.append(count)

    return entities, reference_count


#eds = parse_eds(eds.split('\n'))
#eds_context = parse_eds(context.split('\n'))
#match_pattern(eds, eds_context)
#print(0)
