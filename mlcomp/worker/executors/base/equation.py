import ast
import operator
import pickle
from copy import deepcopy

import cv2
import numpy as np
import albumentations as A

from torch.utils.data import Dataset

from mlcomp.db.providers import ModelProvider
from mlcomp.utils.config import parse_albu_short, Config
from mlcomp.utils.torch import infer
from mlcomp.worker.executors import Executor
from mlcomp.worker.executors.base.tta_wrap import TtaWrap

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
    def __init__(
        self,
        model_id: int,
        suffix: str = '',
        max_count=None,
        part_size: int = None,
        **kwargs
    ):
        self.__dict__.update(kwargs)
        self.model_id = model_id
        name = kwargs.get('name')
        if not name:
            self.name = ModelProvider(self.session).by_id(self.model_id).name

        self.suffix = suffix
        self.max_count = max_count
        self.part_size = part_size
        self.part = None
        self.cache = dict()

    def tta(self, x: Dataset, tfms=()):
        x = deepcopy(x)
        transforms = getattr(x, 'transforms')
        if not transforms:
            return x
        assert isinstance(transforms, A.Compose), \
            'only Albumentations transforms are supported'
        index = len(transforms.transforms)
        for i, t in enumerate(transforms.transforms):
            if isinstance(t, A.Normalize):
                index = i
                break
        tfms_albu = []
        for i, t in enumerate(tfms):
            t = parse_albu_short(t, always_apply=True)
            tfms_albu.append(t)
            transforms.transforms.insert(index + i, t)
        return TtaWrap(x, tfms_albu)

    def adjust_part(self, part):
        pass

    def generate_parts(self, count):
        part_size = self.part_size or count
        res = []
        for i in range(0, count, part_size):
            res.append((i, min(count, i + part_size)))
        return res

    def load(self, file: str = None):
        file = file or self.name + f'_{self.suffix}'
        file = f'data/pred/{file}'

        data = pickle.load(open(file, 'rb'))
        data = data[self.part[0]: self.part[1]]

        if isinstance(data, list):
            for row in data:
                if type(row).__module__ == np.__name__:
                    continue
                if isinstance(row, list):
                    for i, c in enumerate(row):
                        row[i] = cv2.imdecode(c, cv2.IMREAD_GRAYSCALE)
            data = np.array(data)

        return data

    def torch(
        self,
        x: Dataset,
        file: str = None,
        batch_size: int = 1,
        use_logistic: bool = True,
        num_workers: int = 1
    ):
        file = file or self.name + '.pth'
        file = f'models/{file}'
        return infer(
            x=x,
            file=file,
            batch_size=batch_size,
            use_logistic=use_logistic,
            num_workers=num_workers
        )

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return _OP_MAP[type(node.op)](left, right)

    def visit_Name(self, node):
        name = node.id
        attr = getattr(self, name, None)
        if attr:
            if isinstance(attr, str):
                res = self._solve(attr)
                self.cache[name] = res
                return res
            return attr
        return str(name)

    def visit_List(self, node):
        return self.get_value(node)

    def visit_Tuple(self, node):
        return self.get_value(node)

    def visit_Num(self, node):
        return node.n

    def visit_Str(self, node):
        return node.s

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_pow(self, node):
        return node

    def visit_NameConstant(self, node):
        return node.value

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
        if t == ast.List:
            res = []
            for e in node.elts:
                res.append(self.get_value(e))
            return res
        if t == ast.Tuple:
            res = []
            for e in node.elts:
                res.append(self.get_value(e))
            return res
        raise Exception(f'Unknown type {t}')

    def visit_Call(self, node):
        name = node.func.id
        f = getattr(self, name)
        if not f:
            raise Exception(f'Equation class does not contain method = {name}')

        args = [self.get_value(a) for a in node.args]
        kwargs = {k.arg: self.get_value(k.value) for k in node.keywords}
        return f(*args, **kwargs)

    def _solve(self, equation):
        if equation is None:
            return None

        equation = str(equation)
        if equation in self.cache:
            return self.cache[equation]

        tree = ast.parse(equation)
        if len(tree.body) == 0:
            return None
        calc = self
        res = calc.visit(tree.body[0])
        self.cache[equation] = res

        return res

    def solve(self, equation, parts):
        for part in parts:
            self.cache = {}
            self.part = part
            self.adjust_part(part)
            yield self._solve(equation)

    @classmethod
    def _from_config(
        cls, executor: dict, config: Config, additional_info: dict
    ):
        return cls(**executor)


__all__ = ['Equation']
