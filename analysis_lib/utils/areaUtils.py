# import numpy as np
import torch
import logging
from scipy.optimize import minimize
import time

from ..analysisNet import AnalysisNet

import numpy as np


class WapperArea(object):
    """
    修改area生成方式
    最小域冲突
    """

    def __init__(self, funcNum):
        self.list = torch.zeros(funcNum, dtype=torch.int8)
        self.list[0] = -1
        self.idxList = 0
        self.sRSign = torch.Tensor().type(torch.int8)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self._updateList()
            while self.checkList():
                self._updateList()
        except:
            raise StopIteration
        return self.output

    def updateIndex(self, areaSign):
        self.sRSign = torch.cat([self.sRSign, areaSign.view(1, -1)], dim=0)

    def checkList(self):
        try:
            a = ((self.sRSign.abs() * self.output) - self.sRSign).abs().sum(dim=1)
            return (0 in a)
        except:
            return False

    def _updateList(self):
        self.list[0] += 1
        for i in range(self.idxList + 1):
            if self.list[i] == 2:
                self.list[i] = 0
                self.list[i+1] += 1
                if i == self.idxList:
                    self.idxList += 1
        self.output = self.list * 2 - 1


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

        To minimize: (bound - r >= 0)
        * min_{x,r} (bound-r)
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
                return self.bound - x[-1]
            return xx

        def jacCcfunc(function):
            def xx(x):
                output = np.zeros_like(function)
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

        function = ccfunc(funcList[0])
        res = minimize(function, point, method='SLSQP', constraints=cons, jac=jacCcfunc(funcList[0]), tol=1e-10, options={"maxiter": 100})
        x, r = res.x[:-1], res.x[-1]
        return x, r

    def _calulatePoint(self, cFuncList, cArea, pFuncList, pArea, point, layerNum):
        """
        并且简化区域，并且得到一个计算点，如果(aX + b)最小值为0，此超平面就是边界。

        *   min_{x} (aX + b);
        *   s.t. AX + B >= 0;
        """
        funcList, area = torch.cat([pFuncList, cFuncList], dim=0), torch.cat([pArea, cArea], dim=0)
        conFuncs = area.view(-1, 1) * funcList
        funcList, conFuncs, area = funcList.numpy(), conFuncs.numpy(), area.numpy()
        conArea = np.ones_like(area)
        nextFuncList, nextArea, cons = [], [], []
        for i in range(conFuncs.shape[0]):
            con = {
                'type': 'ineq',
                'fun': conFunc(conFuncs[i]),
                'jac': conJac(conFuncs[i]),
            }
            cons.append(con)
        NewPoint = None
        # ==================================================
        # 改用计算多边形最小直径
        if layerNum == 0:
            point = np.random.uniform(-self.bound, self.bound, [conFuncs[0].shape[0], ])
        else:
            r = np.random.uniform(-self.bound, self.bound)
            point = np.append(point, r)
        x, r = self._calulateR(conFuncs, point)
        result = np.matmul(conFuncs[:, :-1], x.T) + conFuncs[:, -1]
        result = np.where(result >= -1e-10, 1, 0)
        if np.array_equal(result, conArea):
            self.logger.info(f"-----------point Layer: {layerNum}--------------")
            # self.logger.info(f"New Point: {x}")
            self.logger.info(f"Distance: {r}")
            NewPoint = x
        # ==================================================
        if NewPoint is None:
            return None, None, None, False, None
        # 通过优化点优化最简区域
        cAreaSign = torch.zeros_like(cArea)
        for i in range(conFuncs.shape[0]):
            function = conFunc(conFuncs[i])
            res = minimize(function, NewPoint, method='SLSQP', constraints=cons, jac=conJac(conFuncs[i]), tol=1e-10, options={"maxiter": 100})
            if res.fun <= 1e-9:
                nextFuncList.append(torch.from_numpy(funcList[i]))
                nextArea.append(area[i])
            if i >= pArea.shape[0]:
                cAreaSign[i-pArea.shape[0]] = area[i]
        nextFuncList = torch.stack(nextFuncList)
        nextArea = torch.tensor(nextArea, dtype=torch.int8)
        cAreaSign = torch.tensor(cAreaSign, dtype=torch.int8)
        nextPoint = NewPoint
        self.logger.info(f"Function size: {nextFuncList.size()};")
        return nextPoint, nextFuncList, nextArea, True, cAreaSign

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
        # 判断每个函数是否穿过区域
        for i in range(funcList.shape[0]):
            function = func(funcList[i])
            res = minimize(function, point, method='SLSQP', constraints=cons, jac=funcJac(funcList[i]), tol=1e-10)
            if res.fun <= 1e-9:
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
            count = 0
            for cArea in layerAreas:
                count += 1
                # print(f"LayerNum: {layerNum}, count: {count}")
                nextPoint, nextFuncList, nextArea, isExist, cAreaSign = self._calulatePoint(cFuncList, cArea, pFuncList, pArea, point, layerNum)
                if not isExist:
                    continue
                layerAreas.updateIndex(cAreaSign)
                areaNum += self._wapperGetLayerAreaNum(nextPoint, nextFuncList, nextArea, layerNum)
        return areaNum

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
            * -1: aX+b < 0;
            *  1: aX+b >= 0;

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
        if (pFuncList is not None) and (pArea is not None):
            point = self._calulatePoint(pFuncList, pArea)
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


def conJac(function):
    """ {a_i;i in (0, n)} """
    def xx(x):
        return function[:-1]
    return xx
