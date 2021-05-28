# import numpy as np
import numpy as np
from ..analysisNet import AnalysisNet
from torch.onnx.symbolic_opset9 import dim
import sys
from typing import Iterable
from numpy.core.fromnumeric import size
from scipy.integrate._ivp.radau import C
import torch
import logging
from scipy.optimize import minimize, Bounds
import time
from collections import deque


class WapperArea(object):
    """
    修改area生成方式;
    最小域冲突;
    基于最小域的搜索方案;
    """

    def __init__(self, funcNum):
        self.list = torch.zeros(funcNum, dtype=torch.int8)
        self.list[0] = -1
        self.idxList = 0
        self.sRSign = torch.Tensor().type(torch.int8)
        self.area = deque()
        # deque(maxlen=self.videoFps*2)
        self.areaIdx = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self._updateList()
            while self.checkArea(self.output):
                self._updateList()
        except:
            raise StopIteration
        return self.output

    def updateIndex(self, areaSign):
        if self.checkArea(areaSign):
            return
        self.sRSign = torch.cat([self.sRSign, areaSign.view(1, -1)], dim=0)

    def registArea(self, areaSign):
        if not self.checkArea(areaSign):
            self.area.append(areaSign)

    def registAreas(self, areaSigns: torch.Tensor):
        for area in areaSigns:
            if not self.checkArea(area):
                self.area.append(area)

    def checkArea(self, area: torch.Tensor):
        try:
            a = ((self.sRSign.abs() * area) - self.sRSign).abs().sum(dim=1)
            return (0 in a)
        except:
            return False

    def _updateList(self):
        self.output = self.area.popleft()


class AnalysisReLUNetUtils(object):
    """
    Arg :
        - device : GPU or CPU to get the graph;
        - logger : print the information (Default: print in console);
    """

    def __init__(self, device=torch.device('cpu'), logger=None):
        self.device = device
        if logger is None:
            self.logger = logging.getLogger("AnalysisReLUNetUtils")
            self.logger.setLevel(level=logging.INFO)  # 定义过滤级别
            formatter = logging.Formatter('[%(asctime)s] - %(name)s : %(message)s')
            console = logging.StreamHandler()  # 日志信息显示在终端terminal
            console.setLevel(logging.INFO)
            console.setFormatter(formatter)
            self.logger.addHandler(console)
        else:
            self.logger = logger

    def _getFuncList(self, point, layerNum):
        """
        得到函数ReLU之前的一层\n
        TODO: 修改其他地方
        """
        self.net.eval()
        point = torch.from_numpy(point).float()
        point = point.to(self.device).unsqueeze(dim=0)
        with torch.no_grad():
            _, weight_graph, bias_graph = self.net.forward_graph_Layer(point, Layer=layerNum)
            # (1, *output.size(), *input.size())
            weight_graph, bias_graph = weight_graph[0], bias_graph[0]
            # (output.num, input.num)
            weight_graph = weight_graph.reshape(-1, self.net.size_Prod)
            # self.logger.info(weight_graph.size())
            bias_graph = bias_graph.sum(dim=list(range(len(bias_graph.shape)))[-len(self.net._input_size):])
            # (output.num, 1)
            bias_graph = bias_graph.reshape(-1, 1)
            # self.logger.info(bias_graph.size())
            # (output.num, input.num + 1)
        return torch.cat([weight_graph, bias_graph], dim=1)

    def _calulateR(self, funcList, point):
        """
        计算多边形中的最小圆的直径，以及得到区域内一点，使用了论文[1]中的方法。:
        * max_{x,r} r
        * s.t.  (AX+B-r||A|| >= 0)

        To minimize: (-r)
        * min_{x,r} (-r)
        * s.t.  (AX+B-r||A|| >= 0)

        [1] Zhang, X. , and  D. Wu . "Empirical Studies on the Properties of Linear Regions in Deep Neural Networks." (2020).
        """
        def cfunc(function, normA):
            def xx(x):
                return np.matmul(x[:-1], function[:-1]) + function[-1] - normA * x[-1]
            return xx

        def jacCfunc(function, normA):
            def xx(x):
                return np.append(function[:-1], -normA)
            return xx

        def ccfunc(funcion):
            def xx(x):
                # return -np.log(x[-1])
                return -x[-1]
            return xx

        def jacCcfunc(function):
            def xx(x):
                output = np.zeros_like(function)
                # output[-1] -= (1 / x[-1])
                output[-1] -= 1
                return output
            return xx

        cons = []
        normA = np.linalg.norm(funcList[:, :-1], ord=2, axis=1)
        for i in range(funcList.shape[0]):
            con = {
                'type': 'ineq',
                'fun': cfunc(funcList[i], normA[i]),
                'jac': jacCfunc(funcList[i], normA[i]),
            }
            cons.append(con)
        cons.extend(self.con)
        function = ccfunc(funcList[0])
        res = minimize(function, point, method='SLSQP', constraints=cons, jac=jacCcfunc(funcList[0]), tol=1e-10, options={"maxiter": 100})
        x, r = res.x[:-1], res.x[-1]
        return x, r

    def _calulateAreaPoint(self, cFuncList, cArea, pFuncList, pArea, point, layerNum):
        """
        并且简化区域，并且得到一个计算点，如果(aX + b)最小值为0，此超平面就是边界。

        *   min_{x} (aX + b);
        *   s.t. AX + B >= 0;
        """
        funcList, area = torch.cat([cFuncList, pFuncList], dim=0), torch.cat([cArea, pArea], dim=0)
        conFuncs = area.view(-1, 1) * funcList
        funcList, conFuncs, area = funcList.numpy(), conFuncs.numpy(), area.numpy()
        conArea = np.ones_like(area)
        nextFuncList, nextArea, cons, funcPoints = [], [], [], []
        NewPoint = None
        # ==================================================
        # 改用计算多边形最小直径
        r = np.random.uniform(0, self.bound)
        if layerNum == 0:
            point = np.random.uniform(-self.bound, self.bound, [conFuncs[0].shape[0] - 1, ])
        point = np.append(point, r)
        x, r = self._calulateR(conFuncs, point)
        result = np.matmul(conFuncs[:, :-1], x.T) + conFuncs[:, -1]
        result = np.where(result >= -1e-10, 1, 0)
        if np.array_equal(result, conArea):
            self.logger.info(f"-----------point Layer: {layerNum}--------------")
            self.logger.info(f"Distance: {r}, x: {x}")
            NewPoint = x
        # ==================================================
        if NewPoint is None:
            return None, None, None, None, None, False

        for i in range(conFuncs.shape[0]):
            con = {
                'type': 'ineq',
                'fun': conFunc1(conFuncs[i]),
                'jac': conJac(conFuncs[i]),
            }
            cons.append(con)
        # 通过优化点优化最简区域
        cons.extend(self.con)
        cAreaSign = torch.zeros_like(cArea)
        for i in range(conFuncs.shape[0]):
            function = func(conFuncs[i])
            cons[i]['fun'] = conFunc(conFuncs[i])
            res = minimize(function, NewPoint, method='SLSQP', constraints=cons, jac=funcJac(conFuncs[i]), tol=1e-20, options={"maxiter": 100})
            if np.abs(res.fun) < 1e-15:
                nextFuncList.append(torch.from_numpy(funcList[i]))
                nextArea.append(area[i])
                # 寻找相邻域
                # print(i, cArea.shape[0], res.fun)
                if i < cArea.shape[0]:
                    cAreaSign[i] = area[i]
                    x, a, b = res.x, conFuncs[i, :-1], conFuncs[i, -1]
                    # -------------------------------
                    # f = cFuncList.numpy()
                    # ress = (np.matmul(f[:, :-1], x) + f[:, -1]) * cArea.numpy()
                    # print(ress.tolist())
                    # -------------------------------
                    if res.fun < 0:
                        k = 0
                    elif res.fun > 0:
                        k = -2 * (np.matmul(a, x) + b + 2e-16) / np.matmul(a, a)
                    else:
                        k = -(np.matmul(a, x) + b + 2e-16) / np.matmul(a, a)
                    x_p = x + k * a
                    funcPoints.append(torch.from_numpy(x_p))
        nextFuncList = torch.stack(nextFuncList)
        funcPoints = torch.stack(funcPoints)
        nextArea = torch.tensor(nextArea, dtype=torch.int8)
        nextPoint = NewPoint
        # self.logger.info(f"Function size: {nextFuncList.size()};")
        return nextPoint, nextFuncList, nextArea, cAreaSign.type(torch.int8), funcPoints, True

    def _calculateFunc(self, funcList, pFuncList, pArea, point):
        """
        计算是否与区域有交点, 目标函数有一点在区域内:
        *    min(func(x)^2);
        *    s.t. pFunc(x) >= 0
        """
        conFuncs = pArea.view(-1, 1) * pFuncList
        conFuncs, funcList = conFuncs.numpy(), funcList.numpy()
        funcs, cons, points = [], [], []
        # 拼接限制条件
        for i in range(conFuncs.shape[0]):
            con = {
                'type': 'ineq',
                'fun': conFunc(conFuncs[i]),
                'jac': conJac(conFuncs[i]),
            }
            cons.append(con)
        cons.extend(self.con)
        # 判断每个函数是否穿过区域
        for i in range(funcList.shape[0]):
            function = func(funcList[i])
            res = minimize(function, point, method='SLSQP', constraints=cons, jac=funcJac(funcList[i]), tol=1e-20, options={"maxiter": 100})
            if res.fun <= 1e-16:
                funcs.append(torch.from_numpy(funcList[i]))
                points.append(torch.from_numpy(res.x))
        if len(funcs) != 0:
            funcs = torch.stack(funcs, dim=0)
            points = torch.stack(points, dim=0)
            self.logger.info("-----------functions--------------")
            self.logger.info(f"Function size: {funcs.size()};")
        else:
            funcs, points = None, None
        return funcs, points

    def _getLayerArea(self, cFuncList, pFuncList, pArea, layerNum, point):
        """
        1. 验证切割的超平面在区域中的存在性:
            a. 不存在 -> 是否是最后一层:
                i.  是 -> areaNum + 1;
                ii. 不是 -> 将pFuncList以及point直接传入下一层进行递归;
            b. 存在 -> 计算每个区域子区域是否存在:
                (1). 不存在: 跳过;
                (2). 存在: -> 是否是最后一层:
                    i.  是: -> areaNum + 1;
                    ii. 不是: -> 将区域拼接, 递归入getNetAreaNum  + areaNum;
        """
        areaNum = 0
        cFuncList, points = self._calculateFunc(cFuncList, pFuncList, pArea, point)
        if cFuncList is None:
            areaNum += self._wapperGetLayerAreaNum(point, pFuncList, pArea, layerNum)
        else:
            layerAreas = WapperArea(cFuncList.shape[0])
            pointAreas = self._getAreaFromPoint(points, cFuncList)
            layerAreas.registAreas(pointAreas)
            for cArea in layerAreas:
                nextPoint, nextFuncList, nextArea, cAreaSign, funcPoints, isExist = self._calulateAreaPoint(cFuncList, cArea, pFuncList, pArea, point, layerNum)
                if not isExist and (nextFuncList is None):
                    continue
                layerAreas.updateIndex(cAreaSign)
                pointAreas = self._getAreaFromPoint(funcPoints, cFuncList)
                layerAreas.registAreas(pointAreas)
                areaNum += self._wapperGetLayerAreaNum(nextPoint, nextFuncList, nextArea, layerNum)
        return areaNum

    def _getAreaFromPoint(self, points: torch.Tensor, funcList: torch.Tensor):
        a, b = funcList[:, :-1].double(), funcList[:, -1].double()
        res = torch.sign(torch.matmul(points, a.T) + b)
        res = torch.where(res == 0, 1., res).type(torch.int8)
        return res

    def _wapperGetLayerAreaNum(self, point, funcList, area, layerNum):
        isLast = (layerNum == self.countLayers)
        nextLayerNum = 1 + layerNum
        if isLast:
            num = 1
            self.regist(point, funcList, area)
        else:
            num = self._getLayerAreaNum(point, funcList, area, nextLayerNum)
        return num

    def _getLayerAreaNum(self, point, pfuncList, pArea, layerNum):
        """
        计算层的区域数量，用递归下一层，直至此区域的最后一层，统计区域数据。
        """
        # 通过点得到函数
        cFuncList = self._getFuncList(point, layerNum).cpu()
        # 计算区域数量
        layerAreaNum = self._getLayerArea(cFuncList, pfuncList, pArea, layerNum, point)
        return layerAreaNum

    def getAreaNum(self, net: AnalysisNet, bound: float = 1.0, countLayers: int = -1, pFuncList: torch.Tensor = None, pArea: torch.Tensor = None, saveArea: bool = False):
        """
        TODO: 目前只支持方形的输入空间画图，需要修改。

        Area:
            *  1: aX+b >= 0;
            * -1: aX+b < 0;

        Function: a tensor (m)
            * a: tensor[ : -1];
            * b: tensor[-1];

        FuncList: a tensor (n x m)
            * A: tensor[ :, : -1];
            * B: tensor[ :, -1];
        """
        assert isinstance(net, AnalysisNet), "the type of net must be \"AnalysisNet\"."
        assert countLayers != -1, "countLayers must >= 0."
        assert bound is not None, "Please set range."
        self.logger.info("Start Get region number...")
        self.net, self.countLayers, self.bound, self.saveArea = net.to(self.device), countLayers, bound, saveArea
        self.regist = self._updateAreaListRegist if saveArea else self._defaultRegist()
        self.areaFuncs, self.areas, self.points = [], [], []
        self.initCon(self.net.size_Prod)
        if (pFuncList is not None) and (pArea is not None):
            point = self._calulateAreaPoint(pFuncList, pArea)
        else:
            pFuncList1 = torch.cat([torch.eye(self.net.size_Prod), torch.zeros(self.net.size_Prod, 1)-self.bound], dim=1)  # < 0
            pFuncList2 = torch.cat([torch.eye(self.net.size_Prod), torch.zeros(self.net.size_Prod, 1)+self.bound], dim=1)  # >=0
            pFuncList = torch.cat([pFuncList1, pFuncList2], dim=0)
            pArea = torch.ones(self.net.size_Prod*2, dtype=torch.int8)
            pArea[0:self.net.size_Prod] = -1
            point = np.zeros([*self.net._input_size])
        return self._getLayerAreaNum(point, pFuncList, pArea, 0)

    def _defaultRegist(self, point, funcList, area):
        return

    def _updateAreaListRegist(self, point, funcList, area):
        self.areaFuncs.append(funcList)
        self.areas.append(area)
        self.points.append(point)

    def initCon(self, num):
        self.con = []
        con = {
            'type': 'ineq',
            'fun': boundFun,
            'jac': boundJac,
        }
        self.con.append(con)

    def getAreaData(self):
        """
        Return:
            areaFuncs: [tensor,];
            areas: [tensor,];
            points: [np.array,];
        """
        assert self.saveArea, "Not save some area infomation"
        return self.areaFuncs, self.areas, self.points


def func(function):
    """  (aX+b)^2 """
    def xx(x):
        return np.square(np.matmul(x, function[:-1]) + function[-1])
    return xx


def funcJac(function):
    """ {2a_i(aX+b);i in (0, n)} """
    def xx(x):
        return np.array([((np.matmul(x, function[:-1]) + function[-1]) * function[i] * 2) for i in range(function.shape[0]-1)])
    return xx


def conFunc(function):
    """ aX+b """
    def xx(x):
        return np.matmul(x, function[:-1]) + function[-1]
    return xx


def conFunc1(function):
    """ aX+b """
    def xx(x):
        return np.matmul(x, function[:-1]) + function[-1] - 1e-8
    return xx


def conJac(function):
    """ {a_i;i in (0, n)} """
    def xx(x):
        return function[:-1]
    return xx


def boundFun(x):
    return np.sum(np.square(x)) - 1e-32


def boundJac(x):
    return np.array([2*x[i] for i in range(x.shape[0])])
