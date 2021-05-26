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

    def _calulatePoint(self, cFuncList, cArea, pFuncList, pArea, point, layerNum):
        """
        并且简化区域, 并且得到一个计算点
         1. 区域是否存在; 简化区域;
        2. 从简化区域中得到point;
        """
        print("-----------point--------------")
        funcList, area = torch.cat([pFuncList, cFuncList], dim=0), torch.cat([pArea, cArea], dim=0)
        # print('pArea Len: ', pArea.size(0))
        # print('area: ', area)
        conFuncs = (area * 2 - 1).view(-1, 1) * funcList
        conArea = torch.ones_like(area)
        nextFuncList, nextArea, cons = [], [], []
        for i in range(conFuncs.size(0)):
            con = {
                'type': 'ineq',
                'fun': func(conFuncs[i]),
                'jac': conJac(conFuncs[i]),
            }
            cons.append(con)
        # 找到优化点
        NewPoint = None
        for i in range(funcList.size(0)):
            if layerNum == 0:
                point = torch.empty([funcList[i].size(0) - 1, ], device=self.device).uniform_(-self.xrange, self.xrange)
            function = func(funcList[i])
            res = minimize(function, point, method='SLSQP', constraints=cons, jac=funcJac(funcList[i]), tol=1e-10, options={"maxiter": 100})
            # 拿到x, x带入, 是否满足所有条件。若不满足，区域不存在。
            result = torch.matmul(conFuncs[:, :-1], res.x.T) + conFuncs[:, -1]
            result = torch.where(result >= 0, 1, 0)
            if torch.equal(result, conArea):
                print("New Point:", res.x)
                NewPoint = res.x
                break
        if NewPoint is None:
            return None, None, None, False
        # 通过优化点优化最简区域
        nextPoint = 0
        for i in range(funcList.size(0)):
            function = func(funcList[i])
            res = minimize(function, NewPoint, method='SLSQP', constraints=cons, jac=funcJac(funcList[i]), tol=1e-10, options={"maxiter": 3000})
            if res.fun <= 1e-4:
                # print(f"Function: {funcList[i]}; Success; fun: {res.fun}; point: {res.x}; statue: {res.status}")
                nextFuncList.append(funcList[i])
                nextArea.append(area[i])
                nextPoint += res.x
        # print(len(nextFuncList))
        nextFuncList = torch.stack(nextFuncList, dim=0)
        nextArea = torch.tensor(nextArea, dtype=torch.int8)
        nextPoint /= nextFuncList.size(0)
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
        funcs, cons = [], []
        # 拼接限制条件
        for i in range(conFuncs.size(0)):
            con = {
                'type': 'ineq',
                'fun': func(conFuncs[i]),
                'jac': conJac(conFuncs[i]),
            }
            cons.append(con)
        # 判断每个函数是否穿过区域
        for i in range(funcList.size(0)):
            function = func(funcList[i])
            res = minimize(function, point, method='SLSQP', constraints=cons, jac=funcJac(funcList[i]), tol=1e-10)
            if res.fun <= 1e-9:
                print(f"Function: {funcList[i]}; Success; fun: {res.fun}; point: {res.x}.")
                funcs.append(funcList[i])
        if len(funcs) != 0:
            funcs = torch.stack(funcs, dim=0)
        else:
            funcs = None
        return funcs

    def _getLayerAreaNum(self, cFuncList, pFuncList, pArea, layerNum, point):
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
        isLast = (layerNum == self.countLayers)
        nextLayerNum = 1 + layerNum
        if cFuncList.size(0) == 0:
            num = 1 if isLast else self._getNetAreaNum(point, pFuncList, pArea, nextLayerNum)
            areaNum += num
        else:
            cFuncList = self._calculateFunc(cFuncList, pFuncList, pArea, point)
            if cFuncList is None:
                num = 1 if isLast else self._getNetAreaNum(point, pFuncList, pArea, nextLayerNum)
                areaNum += num
            else:
                layerAreas = WapperArea(cFuncList.size(0))
                for cArea in layerAreas:
                    # 组合线性区域
                    nextPoint, nextFuncList, nextArea, isExist = self._calulatePoint(cFuncList, cArea, pFuncList, pArea, point, layerNum)
                    if not isExist:
                        continue
                    num = 1 if isLast else self._getNetAreaNum(nextPoint, nextFuncList, nextArea, nextLayerNum)
                    areaNum += num
        return areaNum

    def _getNetAreaNum(self, point, pfuncList, pArea, layerNum):
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
        """
        # 通过点得到函数
        # print(f"Exc the layer: {layerNum}")
        cFuncList = self._getFuncList(point, layerNum).cpu()
        # 计算区域数量
        layerAreaNum = self._getLayerAreaNum(cFuncList, pfuncList, pArea, layerNum, point)
        return layerAreaNum

    def getAreaNum(self, net, xrange=1, countLayers=-1, pFuncList=None, pArea=None):
        assert isinstance(net, AnalysisNet), "the type of net must be \"AnalysisNet\"."
        print("Start Get region number...")
        self.net = net
        self.countLayers = self.net._layer_num if countLayers == -1 else countLayers
        assert xrange is not None, "Please set range."
        self.xrange = xrange
        if pArea is not None:
            pFuncList, pArea = pFuncList, pArea
            point = self._calulatePoint(self.pfuncList, self.pArea)
        else:
            pFuncList1 = torch.cat([torch.eye(self.net.size_Prod), torch.zeros(self.net.size_Prod, 1)+self.xrange], dim=1)  # >=0
            pFuncList2 = torch.cat([torch.eye(self.net.size_Prod), torch.zeros(self.net.size_Prod, 1)-self.xrange], dim=1)  # < 0
            pFuncList = torch.cat([pFuncList1, pFuncList2], dim=0).to(self.device)
            pArea = torch.zeros(self.net.size_Prod*2, dtype=torch.int8, device=self.device)
            pArea[0:self.net.size_Prod] = 1
            point = torch.zeros([*self.net._input_size], device=self.device)
        return self._getNetAreaNum(point, pFuncList, pArea, 0)


def func(function):
    def xx(x):
        return torch.square(torch.matmul(x, function[:-1]) + function[-1])
    return xx


def conJac(function):
    def xx(x):
        return function[:-1]
    return xx


def funcJac(function):
    def xx(x):
        return torch.array([((torch.matmul(x, function[:-1]) + function[-1]) * function[i] * 2) for i in range(function.size(0)-1)])
    return xx
