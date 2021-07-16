import logging
from collections import deque

import numpy as np
import torch
from scipy.optimize import minimize

from torchays.modules.base import AysBaseModule


class WapperArea(object):
    """
    Get the area(sign) of the funtion list.
    *  1 : f(x) >= 0
    * -1 : f(x) < 0
    """

    def __init__(self):
        self.sRSign = torch.Tensor().type(torch.int8)
        self.area = deque()

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

    def _updateList(self):
        self.output = self.area.popleft()

    def registArea(self, areaSign):
        if not self.checkArea(areaSign):
            self.area.append(areaSign)

    def registAreas(self, areaSigns: torch.Tensor):
        for area in areaSigns:
            self.registArea(area)

    def updateIndex(self, areaSign):
        if self.checkArea(areaSign):
            return
        self.sRSign = torch.cat([self.sRSign, areaSign.view(1, -1)], dim=0)

    def checkArea(self, area: torch.Tensor):
        try:
            a = ((self.sRSign.abs() * area) - self.sRSign).abs().sum(dim=1)
            return (0 in a)
        except:
            return False


class AnalysisReLUNetUtils(object):
    """
    AnalysisReLUNetUtils.

    args:
        device: torch.device
            GPU or CPU to get the graph;
        logger: def info(...)
            print the information (Default: print in console)(logger.info(...)).
    """

    def __init__(self, device=torch.device('cpu'), logger=None):
        self.device = device
        self.one = torch.ones(1, device=self.device).double()
        if logger is None:
            self.logger = logging.getLogger("AnalysisReLUNetUtils-Console")
            self.logger.setLevel(level=logging.INFO)
            formatter = logging.Formatter('[%(asctime)s] - %(name)s : %(message)s')
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(formatter)
            self.logger.addHandler(console)
        else:
            self.logger = logger

    def _getFuncList(self, point, layerNum):
        """
        Get the list of the linear function before ReLU layer.
        """
        point = torch.from_numpy(point).float()
        point = point.to(self.device).unsqueeze(dim=0)
        with torch.no_grad():
            _, graph = self.net.forward_graph_Layer(point, layer=layerNum)
            # (1, *output.size(), *input.size())
            weight_graph, bias_graph = graph['weight_graph'], graph['bias_graph']
            # (output.num, input.num)
            weight_graph = weight_graph.reshape(-1, point.size()[1:].numel())
            # self.logger.info(weight_graph.size())
            # (output.num, 1)
            bias_graph = bias_graph.reshape(-1, 1)
            # self.logger.info(bias_graph.size())
            # (output.num, input.num + 1)
        return torch.cat([weight_graph, bias_graph], dim=1)

    def _calulateR(self, funcList, point):
        """
        Calculate the max radius of the insphere in the region to make the radius less than the distance of the insphere center to the all functions
        which can express a liner region. 

        maximizing the following function.
        * max_{x,r} r
        * s.t.  (AX+B-r||A|| >= 0)

        To minimize: 
        * min_{x,r} (-r)
        * s.t.  (AX+B-r||A|| >= 0)
        """
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
        *   min_{x} (aX + b);
        *   s.t. AX + B >= 0;
        """
        funcList, area = torch.cat([cFuncList, pFuncList], dim=0), torch.cat([cArea, pArea], dim=0)
        conFuncs = area.view(-1, 1) * funcList
        funcList, conFuncs, area = funcList.numpy(), conFuncs.numpy(), area.numpy()
        conArea = np.ones_like(area)
        nextFuncList, nextArea, funcPoints = [], [], []
        # ==================================================
        r = np.random.uniform(0, self.bound)
        if layerNum == 0:
            point = np.random.uniform(-self.bound, self.bound, [conFuncs[0].shape[0] - 1, ])
        point = np.append(point, r)
        NewPoint, r = self._calulateR(conFuncs, point)
        result = np.matmul(conFuncs[:, :-1], NewPoint.T) + conFuncs[:, -1]
        result = np.where(result >= -1e-10, 1, 0)
        if not np.array_equal(result, conArea) or r < 10e-7 or r > self.bound:
            return None, None, None, None, None, False,
        self.logger.info(f"-----------point Layer: {layerNum}--------------")
        self.logger.info(f"Distance: {r}, Point: {NewPoint}")
        # ==================================================
        # Find the least linear functions to express a region.
        cons = [{'type': 'ineq',
                 'fun': conFunc1(conFuncs[i]),
                 'jac': conJac(conFuncs[i]),
                 } for i in range(conFuncs.shape[0])]
        cons.extend(self.con)
        cAreaSign = torch.zeros_like(cArea)
        for i in range(conFuncs.shape[0]):
            function = func(conFuncs[i])
            cons[i]['fun'] = conFunc(conFuncs[i])
            res = minimize(function, NewPoint, method='SLSQP', constraints=cons, jac=funcJac(conFuncs[i]), tol=1e-20, options={"maxiter": 100})
            # print(i, res.fun, cArea.shape[0])
            if res.fun > 1e-15:
                continue
            nextFuncList.append(torch.from_numpy(funcList[i]))
            nextArea.append(area[i])
            # Find the points of neighbor area(rigon).
            if i < cArea.shape[0]:
                cAreaSign[i] = area[i]
                x, a, b = res.x, conFuncs[i, :-1], conFuncs[i, -1]
                if res.fun > 0:
                    k = -2 * (np.matmul(a, x) + b) / (np.matmul(a, a))
                    k2 = -2 * (np.matmul(a, x) + b + 1e-10) / (np.matmul(a, a))
                else:
                    k = -(np.matmul(a, x) + b + 1e-10) / (np.matmul(a, a))
                    k2 = -2 * (np.matmul(a, x) + b + 1e-15) / (np.matmul(a, a))
                x_p = x + k * a
                x_p2 = x + k2 * a
                funcPoints.append(torch.from_numpy(x_p))
                funcPoints.append(torch.from_numpy(x_p2))
        nextFuncList = torch.stack(nextFuncList)
        funcPoints = torch.stack(funcPoints)
        nextArea = torch.tensor(nextArea, dtype=torch.int8)
        self.logger.info(f"Smallest function size: {nextFuncList.size()};")
        return NewPoint, nextFuncList, nextArea, cAreaSign.type(torch.int8), funcPoints, True

    def _calculateFunc(self, funcList, pFuncList, pArea, point):
        """
        计算是否与区域有交点, 目标函数有一点在区域内:
        *    min(func(x)^2);
        *    s.t. pFunc(x) >= 0
        """
        conFuncs = pArea.view(-1, 1) * pFuncList
        conFuncs, funcList = conFuncs.numpy(), funcList.numpy()
        funcs, points = [], []
        cons = [{
            'type': 'ineq',
                'fun': conFunc(conFuncs[i]),
                'jac': conJac(conFuncs[i]),
                } for i in range(conFuncs.shape[0])]
        cons.extend(self.con)
        # Is the linear function though the area(Region).
        for i in range(funcList.shape[0]):
            function = func(funcList[i])
            res = minimize(function, point, method='SLSQP', constraints=cons, jac=funcJac(funcList[i]), tol=1e-20, options={"maxiter": 100})
            if res.fun > 1e-16:
                continue
            funcs.append(torch.from_numpy(funcList[i]))
            points.append(torch.from_numpy(res.x))
        if len(funcs) != 0:
            funcs = torch.stack(funcs, dim=0)
            points = torch.stack(points, dim=0)
            self.logger.info("-----------functions though region--------------")
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
            layerAreas = WapperArea()
            pointAreas = self._getAreaFromPoint(points, cFuncList)
            # Regist some areas in wapperArea for iterate.
            layerAreas.registAreas(pointAreas)
            for cArea in layerAreas:
                nextPoint, nextFuncList, nextArea, cAreaSign, funcPoints, isExist = self._calulateAreaPoint(cFuncList, cArea, pFuncList, pArea, point, layerNum)
                if not isExist and (nextFuncList is None):
                    continue
                # Add the area to prevent counting again.
                layerAreas.updateIndex(cAreaSign)
                pointAreas = self._getAreaFromPoint(funcPoints, cFuncList)
                # Regist new areas for iterate.
                layerAreas.registAreas(pointAreas)
                areaNum += self._wapperGetLayerAreaNum(nextPoint, nextFuncList, nextArea, layerNum)
        return areaNum

    def _getAreaFromPoint(self, points: torch.Tensor, funcList: torch.Tensor):
        a, b = funcList[:, :-1].double(), funcList[:, -1].double()
        area = torch.sign(torch.matmul(points, a.T) + b)
        area = torch.where(area == 0, self.one, area).type(torch.int8)
        return area

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
        # Get the list of the linear functions for DNN.
        cFuncList = self._getFuncList(point, layerNum).cpu()
        # 更新, 不让其太小
        cFuncList = self._updateFuncList(cFuncList)
        # Get the region number of one layer.
        layerAreaNum = self._getLayerArea(cFuncList, pfuncList, pArea, layerNum, point)
        return layerAreaNum

    def _updateFuncList(self, funcList):
        for i in range(funcList.shape[0]):
            funcList[i] = self._updataFunc(funcList[i])
        return funcList

    def _updataFunc(self, func):
        a, _ = func.abs().max(dim=0)
        if a < 0.001:
            func *= 1000
            return self._updataFunc(func)
        return func

    def getAreaNum(self, net: AysBaseModule,
                   bound: float = 1.0,
                   countLayers: int = -1,
                   inputSize: tuple = (2,),
                   pFuncList: torch.Tensor = None,
                   pArea: torch.Tensor = None,
                   saveArea: bool = False):
        """
        目前只支持方形的输入空间画图，需要修改。
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
        assert isinstance(net, AysBaseModule), "the type of net must be \"AysBaseModule\"."
        assert countLayers != -1, "countLayers must >= 0."
        assert bound > 0, "Please set the bound > 0."
        self.logger.info("Start Get region number...")
        self.net, self.countLayers, self.bound, self.saveArea = net.to(self.device), countLayers, bound, saveArea
        self.net.eval()
        self.regist = self._updateAreaListRegist if saveArea else self._defaultRegist
        self.areaFuncs, self.areas, self.points = [], [], []
        self.initCon()
        if (pFuncList is not None) and (pArea is not None):
            point = self._calulateAreaPoint(pFuncList, pArea)
        else:
            size_prod = torch.Size(inputSize).numel()
            pFuncList1 = torch.cat([torch.eye(size_prod), torch.zeros(size_prod, 1)-self.bound], dim=1)  # < 0
            pFuncList2 = torch.cat([torch.eye(size_prod), torch.zeros(size_prod, 1)+self.bound], dim=1)  # >=0
            pFuncList = torch.cat([pFuncList1, pFuncList2], dim=0)
            pArea = torch.ones(size_prod*2, dtype=torch.int8)
            pArea[0:size_prod] = -1
            point = np.zeros(*inputSize)
        regionNum = self._getLayerAreaNum(point, pFuncList, pArea, 0)
        self.logger.info(f"regionNum: {regionNum}")
        return regionNum

    def initCon(self):
        self.con = [{'type': 'ineq',
                     'fun': boundFun,
                     'jac': boundJac, }]

    def _defaultRegist(self, point, funcList, area):
        return

    def _updateAreaListRegist(self, point, funcList, area):
        self.areaFuncs.append(funcList)
        self.areas.append(area)
        self.points.append(point)

    def getAreaData(self):
        """
        Return:
            areaFuncs: [tensor,];
            areas: [tensor,];
            points: [np.array,];
        """
        assert self.saveArea, "Not save some area infomation"
        return self.areaFuncs, self.areas, self.points

# ================================================================
# Minimize function.


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
        return np.matmul(x, function[:-1]) + function[-1] - 1e-10
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
        return -x[-1]
    return xx


def jacCcfunc(function):
    def xx(x):
        output = np.zeros_like(function)
        output[-1] -= 1
        return output
    return xx
