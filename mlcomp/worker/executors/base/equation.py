import ast
import operator

import numpy as np

from torch.utils.data import Dataset

from mlcomp.utils.torch import infer
from mlcomp.worker.executors import Executor

_OP_MAP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Invert: operator.neg,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow
}


@Executor.register
class Equation(Executor, ast.NodeVisitor):
    # noinspection PyTypeChecker
    def __init__(self, equations: str, target: str):
        self.equations = dict(
            [row.strip().split(':') for row in equations.split('\n') if
             ':' in row])
        self.target = target

    def load(self, file: str, type: str = 'numpy'):
        if type == 'numpy':
            return np.load(file)

        raise Exception(f'Unknown load type = {type}')

    def torch(self, x: Dataset, file: str, batch_size: int = 1,
              use_logistic: bool = True, num_workers: int = 1):
        return infer(x=x, file=file, batch_size=batch_size,
                     use_logistic=use_logistic, num_workers=num_workers)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return _OP_MAP[type(node.op)](left, right)

    def visit_Name(self, node):
        name = node.id
        if name in self.equations:
            return self.solve(name)
        attr = getattr(self, name)
        if attr:
            return attr
        raise Exception(f'Unknown symbol {name}')

    def visit_Num(self, node):
        return node.n

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_pow(self, node):
        return node

    def get_value(self, node):
        t = type(node)
        if t == ast.NameConstant:
            return node.value
        if t == ast.Name:
            return self.visit_Name(node)
        if t == ast.Str:
            return node.s
        if t == ast.Name:
            return node.id
        if t == ast.Num:
            return node.n
        raise Exception(f'Unknown type {t}')

    def visit_Call(self, node):
        name = node.func.id
        f = getattr(self, name)
        if not f:
            raise Exception(f'Equation class does not contain method = {name}')

        args = [self.get_value(a) for a in node.args]
        kwargs = {k.arg: self.get_value(k.value) for k in node.keywords}
        return f(*args, **kwargs)

    def solve(self, name: str):
        equation = self.equations[name]
        tree = ast.parse(equation)
        calc = self
        res = calc.visit(tree.body[0])
        return res

    def work(self) -> dict:
        res = self.solve(self.target)
        return {'res': res}


if __name__ == '__main__':
    eq = Equation(
        '''
a:torch(x, batch_size=24, use_logistic=True)
y:a+30
        ''',
        'y'
    )
    print(eq.work())
