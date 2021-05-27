# import numpy as np
import torch
from scipy.optimize import minimize

from ..analysisNet import AnalysisNet

import numpy as np


class WapperArea(object):
    """ 修改area生成方式 """

    def __init__(self, funcNum):
        self.list = torch.zeros(funcNum, dtype=torch.int8)
        self.list[0] = -1
        self.idxList = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self._updataList()
        except:
            raise StopIteration
        return self.list

    def _updataList(self):
        self.list[0] += 1
        for i in range(self.idxList + 1):
            if self.list[i] == 2:
                self.list[i] = 0
                self.list[i+1] += 1
                if i == self.idxList:
                    self.idxList += 1


class AnalysisReLUNetUtils(object):
    def __init__(self, device=torch.device('cpu')):
        self.device = device

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
            # print(weight_graph.size())
            bias_graph = bias_graph.sum(dim=list(range(len(bias_graph.shape)))[-len(self.net._input_size):])
            # (output.num, 1)
            bias_graph = bias_graph.reshape(-1, 1)
            # print(bias_graph.size())
            # (output.num, input.num + 1)
        return torch.cat([weight_graph, bias_graph], dim=1)

    def _calulateR(self, funcList, point):
        """ 
        计算多边形中的最小圆的直径:
        * max_{x,r} r  -> min_{x,r} (bound-r)
        * s.t.  (AX+B-r||A|| >= 0)
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
                return self.xrange - x[-1]
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
        并且简化区域, 并且得到一个计算点.\n
        TODO: 计算多边形中的最小圆的直径:
        * max_{x,r} r  -> min_{x,r} (bound-r)
        * s.t.  (AX+B-r||A|| >= 0)
        *       (x >= -bound + r) -> (x-r+bound >= 0)
        *       (x <= +bound - r) -> (x+r-bound <= 0) -> (-x-r+bound >= 0)
        """
        # print("-----------point--------------")
        funcList, area = torch.cat([pFuncList, cFuncList], dim=0), torch.cat([pArea, cArea], dim=0)
        conFuncs = (area * 2 - 1).view(-1, 1) * funcList
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
        # 改用计算多边形最小直径, 减少循环
        if layerNum == 0:
            point = np.random.uniform(-self.xrange, self.xrange, [conFuncs[0].shape[0], ])
        else:
            r = np.random.uniform(-self.xrange, self.xrange)
            point = np.append(point, r)
        x, r = self._calulateR(conFuncs, point)
        result = np.matmul(conFuncs[:, :-1], x.T) + conFuncs[:, -1]
        result = np.where(result >= -1e-10, 1, 0)
        if np.array_equal(result, conArea):
            print(f"-----------point Layer: {layerNum}--------------")
            print("New Point:", x)
            print("Distance: ", r)
            NewPoint = x
        # ==================================================
        if NewPoint is None:
            return None, None, None, False
        # 通过优化点优化最简区域
        for i in range(conFuncs.shape[0]):
            function = conFunc(conFuncs[i])
            res = minimize(function, NewPoint, method='SLSQP', constraints=cons, jac=conJac(conFuncs[i]), tol=1e-10, options={"maxiter": 100})
            if res.fun <= 1e-9:
                print(f"Function: {funcList[i]}; Success; fun: {res.fun}; point: {res.x}; statue: {res.status}")
                nextFuncList.append(torch.from_numpy(funcList[i]))
                nextArea.append(area[i])
        nextFuncList = torch.stack(nextFuncList, dim=0)
        nextArea = torch.tensor(nextArea, dtype=torch.int8)
        nextPoint = NewPoint
        return nextPoint, nextFuncList, nextArea, True

    def _calculateFunc(self, funcList, pFuncList, pArea, point):
        """
        计算是否与区域有交点, 目标函数有一点在区域内:
        TODO: 添加jac
        *    min(func(x)^2);
        *    s.t. pFunc(x) >= 0
        """
        print("-----------functions--------------")
        conFuncs = (pArea * 2 - 1).view(-1, 1) * pFuncList
        conFuncs, funcList = conFuncs.numpy(), funcList.numpy()
        funcs, cons = [], []
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
                print(f"Function: {funcList[i]}; Success; fun: {res.fun}; point: {res.x}.")
                funcs.append(torch.from_numpy(funcList[i]))
        if len(funcs) != 0:
            funcs = torch.stack(funcs, dim=0)
        else:
            funcs = None
        return funcs

    def _getLayerArea(self, cFuncList, pFuncList, pArea, layerNum, point):
        """
        1. 验证funcList在pFuncList中的存在性。
            a. 存在, 切割,并且划出计算子区域, 以及其中的point, 递归。
            b. 不存在, 将pFuncList以及point直接传入下一层。不用计算
        2. 对比每个区域(layerArea),(layerCompare)是否成立:
            a. 成立: -> 是否是最后一层:
                i. 是: -> areaNum + 1;
                ii. 不是: -> 将区域拼接, 递归入getNetAreaNum  + areaNum;
            b. 不成立: 跳过;

        area:   0: <0;
                1: >= 0;
        """
        areaNum = 0
        cFuncList = self._calculateFunc(cFuncList, pFuncList, pArea, point)
        if cFuncList is None:
            areaNum += self._wapperGetLayerAreaNum(point, pFuncList, pArea, layerNum)
        else:
            layerAreas = WapperArea(cFuncList.shape[0])
            for cArea in layerAreas:
                # 组合线性区域
                nextPoint, nextFuncList, nextArea, isExist = self._calulatePoint(cFuncList, cArea, pFuncList, pArea, point, layerNum)
                if not isExist:
                    continue
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
        1. 从区域中找到合适的点;
        2. 通过点带入神经网络, 得到超平面;
        3. 判断超平面是否存在于区域内 并且 超平面是否相同;
        4. 是否是最后一层:
            ---------是---------
            a. 计算超平面相交的超平面的数量(少一个维度的);
            b. 计算区域数量;
            ---------不是-------
            a. 列举出区域, 并且判断区域是否存在:
                i. 存在: 重复1, 并带入下一层;
                ii.不存在, 选择下一个区域, 重复a;
        5. TODO: 返回区域list, 提供画图工具
        """
        # 通过点得到函数
        cFuncList = self._getFuncList(point, layerNum).cpu()
        # 计算区域数量
        layerAreaNum = self._getLayerArea(cFuncList, pfuncList, pArea, layerNum, point)
        return layerAreaNum

    def getAreaNum(self, net, xrange=1, countLayers=-1, pFuncList=None, pArea=None, saveArea=False):
        assert isinstance(net, AnalysisNet), "the type of net must be \"AnalysisNet\"."
        assert countLayers != -1, "countLayers must >= 0."
        assert xrange is not None, "Please set range."
        print("Start Get region number...")
        self.net, self.countLayers = net, countLayers
        if saveArea:
            self.saveArea = saveArea
            self.regist = self._updateAreaListRegist
            self.areaFuncs, self.areas, self.points = [], [], []
        self.xrange = xrange
        if pArea is not None:
            pFuncList, pArea = pFuncList, pArea
            point = self._calulatePoint(self.pfuncList, self.pArea)
        else:
            pFuncList1 = torch.cat([torch.eye(self.net.size_Prod), torch.zeros(self.net.size_Prod, 1)+self.xrange], dim=1)  # >=0
            pFuncList2 = torch.cat([torch.eye(self.net.size_Prod), torch.zeros(self.net.size_Prod, 1)-self.xrange], dim=1)  # < 0
            pFuncList = torch.cat([pFuncList1, pFuncList2], dim=0)
            pArea = torch.zeros(self.net.size_Prod*2, dtype=torch.int8)
            pArea[0:self.net.size_Prod] = 1
            point = np.zeros([*self.net._input_size])
        return self._getLayerAreaNum(point, pFuncList, pArea, 0)

    def _defaultRegist(self, point, funcList, area):
        return

    def _updateAreaListRegist(self, point, funcList, area):
        self.areaFuncs.append(funcList)
        self.areas.append(area)
        self.points.append(point)

    def getAreaData(self):
        assert self.saveArea, "Not save some area infomation"
        return self.areaFuncs, self.areas, self.points


def func(function):
    def xx(x):
        return np.square(np.matmul(x, function[:-1]) + function[-1])
    return xx


def funcJac(function):
    def xx(x):
        return np.array([((np.matmul(x, function[:-1]) + function[-1]) * function[i] * 2) for i in range(function.shape[0]-1)])
    return xx


def conFunc(function):
    def xx(x):
        return np.matmul(x, function[:-1]) + function[-1]
    return xx


def conJac(function):
    def xx(x):
        return function[:-1]
    return xx
