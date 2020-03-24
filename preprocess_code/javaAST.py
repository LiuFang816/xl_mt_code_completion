import javalang

FuncExample = \
"""private void doParse(String pakageName) throws Exception
{
    if (StringUtils.isBlank(packageName))
    {
        creatorAnnotationParser.parse();
    } else
    {
        creatorAnnotationParser.parse(packageName);
    }
}"""

# 传入一个函数的代码，得到相应的解析树/抽象语法树；也可以不传入，查看上方样例的解析结果
def func2AST(code=None):
    if code is None:
        code = FuncExample
        print("No code/function passed in, function below is to show you a case:")
        print()
        print(code)
        print()
    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parser.Parser(tokens)
    AST = parser.parse_member_declaration()
    return AST

# 辅助打印AST中叶子结点信息的API，做法是将字符串'it\'s a string.'打印成'\'it\'s a string.\''，方便识别和后续处理
def _print_leave_node(depth, indent, content, suffix=None):
    if suffix is None:
        if isinstance(content, str):
            print(depth*indent+"'"+str(content)+"'")
        else:
            print(depth*indent+str(content))
    else:
        if isinstance(content, str):
            print(depth*indent+"'"+str(content)+"'", suffix)
        else:
            print(depth*indent+str(content), suffix)

def showAST_onlyLeafWithValue(ast, depth=0, indent='  |'):
    if depth==0:
        assert isinstance(ast, javalang.ast.Node), 'param <ast> passed in is not an instance of javalang.ast.Node: %s'%type(ast)
        print("<"+str(ast)+">")
        showAST_onlyLeafWithValue(ast, depth+1)
        return

    values_queue = []
    is_leafNode = True

    # DFS show children AST nodes, and saving "values" at the same time.
    for value in ast.children:
        if type(value) in [list, set, tuple]:
            if len(value)==0:
                continue
            for v in value:
                if isinstance(v, javalang.ast.Node):
                    is_leafNode = False
                    print(depth*indent+"<"+str(v)+">")
                    showAST_onlyLeafWithValue(v, depth+1)
                else:
                    values_queue.append(v)
        else:
            if isinstance(value, javalang.ast.Node):
                is_leafNode = False
                print(depth*indent+"<"+str(value)+">")
                showAST_onlyLeafWithValue(value, depth+1)
            else:
                values_queue.append(value)

    # Only show "values" for real leaf AST Nodes.
    if is_leafNode:
        for value in values_queue:
            if value not in [None, '']:
                _print_leave_node(depth, indent, value)

# 将字符串'it\'s a string.'打印成'\'it\'s a string.\''，方便识别和后续处理
def _norm_string(content):
    if isinstance(content, str):
        return "'"+content+"'"
    else:
        return str(content)

def traverseAST_onlyLeafWithValue(ast):
    result = []
    cur_node_type = str(ast)
    result.append(cur_node_type)

    values_queue = []
    is_leafNode = True
    # DFS traverse children AST nodes, and saving "values" at the same time.
    for value in ast.children:
        if type(value) in [list, set, tuple]:
            for v in value:
                if isinstance(v, javalang.ast.Node):
                    is_leafNode = False
                    result += traverseAST_onlyLeafWithValue(v)
                else:
                    values_queue.append(v)
        else:
            if isinstance(value, javalang.ast.Node):
                is_leafNode = False
                result += traverseAST_onlyLeafWithValue(value)
            else:
                values_queue.append(value)

    # "values" for real leaf AST Nodes.
    if is_leafNode:
        for value in values_queue:
            if value not in ['', None]:
                result.append(value)

    return result

def dumpAST_onlyLeafWithValue(ast):

    result = []
    def _dumpAST_onlyLeafWithValue(ast, parent_node_index=-1):

        cur_node_index = len(result)
        cur_node_type = str(ast)
        result.append({"type": cur_node_type, "children": []})
        if parent_node_index!=-1:
            result[parent_node_index]["children"].append(cur_node_index)

        values_queue = []
        is_leafNode = True
        # DFS traverse children AST nodes, and saving "values" at the same time.
        for value in ast.children:
            if type(value) in [list, set, tuple]:
                for v in value:
                    if isinstance(v, javalang.ast.Node):
                        is_leafNode = False
                        _dumpAST_onlyLeafWithValue(v, cur_node_index)
                    else:
                        values_queue.append(v)
            else:
                if isinstance(value, javalang.ast.Node):
                    is_leafNode = False
                    _dumpAST_onlyLeafWithValue(value, cur_node_index)
                else:
                    values_queue.append(value)

        # "values" for real leaf AST Nodes.
        if is_leafNode:
            values_queue = [value for value in values_queue if value not in ['', None]]
            result[cur_node_index]["value"] = values_queue
            del result[cur_node_index]["children"]

    _dumpAST_onlyLeafWithValue(ast)
    return result

if __name__ == "__main__":
    ast = func2AST()
    showAST_onlyLeafWithValue(ast)
    print(traverseAST_onlyLeafWithValue(ast))
    result = dumpAST_onlyLeafWithValue(ast)
    for r in result:
        print(r)