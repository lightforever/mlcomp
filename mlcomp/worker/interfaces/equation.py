import ast
import operator

from os.path import join
import numpy as np

from mlcomp.utils.settings import DATA_FOLDER
from mlcomp.worker.interfaces.base import Interface

_OP_MAP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Invert: operator.neg,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow
}


# noinspection PyPep8Naming,PyMethodMayBeStatic
@Interface.register
class EquationInterface(Interface, ast.NodeVisitor):
    __syn__ = 'equation'

    def __init__(self,
                 project_name: str,
                 suffix: str,
                 equation: str,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.project_name = project_name
        self.equation = equation
        self.suffix = suffix

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return _OP_MAP[type(node.op)](left, right)

    def visit_Name(self, node):
        name = node.id
        data_folder = join(DATA_FOLDER, self.project_name)
        path = join(data_folder, f'{name}_{self.suffix}.npy')
        val = np.load(path)
        return val

    def visit_Num(self, node):
        return node.n

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_pow(self, node):
        return node

    def __call__(self, x: dict) -> dict:
        tree = ast.parse(self.equation)
        calc = self
        prob = calc.visit(tree.body[0])
        return {'prob': prob}


if __name__ == '__main__':
    inf = EquationInterface('examples',
                            'valid', 'a+(1.5*a)+a**(3/4)', name='a')
    inf({})
