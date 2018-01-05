from random import random, randint
from math import sqrt, exp, fabs, inf
from copy import copy

from multilevel import calcDist

def distChangeFromReversal(msr, idx, reversals, pointOrder):
	"""
	Calculates the change of the external distance that comes from
	reversing (or re-reversing) the subroute of index idx.
	"""
	assert idx < len(msr.subRoutes)
	assert idx >= 0

	srLeft = msr.subRoutes[pointOrder[(idx-1 + len(msr.subRoutes)) % len(msr.subRoutes)]]
	srRight = msr.subRoutes[pointOrder[(idx+1) % len(msr.subRoutes)]]

	sr = msr.subRoutes[pointOrder[idx]]

	srLeft2nd = None
	if reversals[pointOrder[(idx-1 + len(msr.subRoutes)) % len(msr.subRoutes)]]:
		srLeft2nd = srLeft.getFirstEndPoint()
	else:
		srLeft2nd = srLeft.getSecondEndPoint()

	srRight1st = None
	if reversals[pointOrder[(idx+1) % len(msr.subRoutes)]]:
		srRight1st = srRight.getSecondEndPoint()
	else:
		srRight1st = srRight.getFirstEndPoint()

	sr1st = sr.getFirstEndPoint()
	sr2nd = sr.getSecondEndPoint()
	if reversals[pointOrder[idx]]:
		tmp = sr1st
		sr1st = sr2nd
		sr2nd = tmp

	oldDist = calcDist(srLeft2nd, sr1st) + calcDist(srRight1st, sr2nd)
	newDist = calcDist(srLeft2nd, sr2nd) + calcDist(srRight1st, sr1st)

	reversals[pointOrder[idx]] = not reversals[pointOrder[idx]]

	return newDist-oldDist

def distChangeFrom2opt(msr, lineIdx1, lineIdx2, isFirstCase):
	# TODO: Check if this is correct even if there's an overlap in points.

	# TODO: This function needs pointOrder!

	distBefore = calcDist(msr.subRoutes[lineIdx1].getSecondEndPoint(), msr.subRoutes[(lineIdx1+1)%len(msr.subRoutes)].getFirstEndPoint()) + calcDist(msr.subRoutes[lineIdx2].getSecondEndPoint(), msr.subRoutes[(lineIdx2+1)%len(msr.subRoutes)].getFirstEndPoint())

	distAfter = None

	if isFirstCase:
		distAfter = calcDist(msr.subRoutes[lineIdx1].getSecondEndPoint(), msr.subRoutes[(lineIdx2+1)%len(msr.subRoutes)].getFirstEndPoint()) + calcDist(msr.subRoutes[lineIdx2].getSecondEndPoint(), msr.subRoutes[(lineIdx1+1)%len(msr.subRoutes)].getFirstEndPoint())
	else:
		distAfter = calcDist(msr.subRoutes[lineIdx1].getSecondEndPoint(), msr.subRoutes[(lineIdx2)%len(msr.subRoutes)].getFirstEndPoint()) + calcDist(msr.subRoutes[lineIdx2+1].getSecondEndPoint(), msr.subRoutes[(lineIdx1+1)%len(msr.subRoutes)].getFirstEndPoint())

	return distAfter - distBefore

def swap2neighSubGroups(msr, idx, newIdx, reversals, pointOrder):
	n = len(msr.subRoutes)
	assert abs(idx-newIdx)==1 or (idx==0 and newIdx==n-1) or (newIdx==0 and idx==n-1)

	if idx==0 and newIdx==n-1:
		leftIdx = newIdx
		rightIdx = idx
	elif idx < newIdx or (idx==n-1 and newIdx==0):
		leftIdx = idx
		rightIdx = newIdx
	else:
		leftIdx = newIdx
		rightIdx = idx


	p1 = msr.subRoutes[pointOrder[leftIdx]]
	pLeft = msr.subRoutes[pointOrder[(leftIdx-1+3*n)%n]]
	p2 = msr.subRoutes[pointOrder[rightIdx]]
	pRight = msr.subRoutes[pointOrder[(rightIdx+1)%n]]

	p1.isReversed = not p1.isReversed if reversals[pointOrder[leftIdx]] else p1.isReversed
	p2.isReversed = not p2.isReversed if reversals[pointOrder[rightIdx]] else p2.isReversed
	pLeft.isReversed = not pLeft.isReversed if reversals[pointOrder[(leftIdx-1+3*n)%n]] else pLeft.isReversed
	pRight.isReversed = not pRight.isReversed if reversals[pointOrder[(rightIdx+1)%n]] else pRight.isReversed

	distBefore = calcDist(pLeft.getSecondEndPoint(), p1.getFirstEndPoint()) + calcDist(p1.getSecondEndPoint(), p2.getFirstEndPoint() ) + calcDist( p2.getSecondEndPoint(), pRight.getFirstEndPoint())

	distAfter = calcDist(pLeft.getSecondEndPoint(), p2.getFirstEndPoint()) + calcDist(p2.getSecondEndPoint(), p1.getFirstEndPoint() ) + calcDist( p1.getSecondEndPoint(), pRight.getFirstEndPoint())

	p1.isReversed = not p1.isReversed if reversals[pointOrder[leftIdx]] else p1.isReversed
	p2.isReversed = not p2.isReversed if reversals[pointOrder[rightIdx]] else p2.isReversed
	pLeft.isReversed = not pLeft.isReversed if reversals[pointOrder[(leftIdx-1+3*n)%n]] else pLeft.isReversed
	pRight.isReversed = not pRight.isReversed if reversals[pointOrder[(rightIdx+1)%n]] else pRight.isReversed

	tmp = pointOrder[idx]
	pointOrder[idx] = pointOrder[newIdx]
	pointOrder[newIdx] = tmp

	return distAfter-distBefore

def moveOneSubGroup(msr, idx, newIdx, reversals, shouldReverse, pointOrder):
	n = len(msr.subRoutes)

	assert n > 3

	if idx == newIdx:
		return 0.0

	if abs(idx-newIdx)==1 or (idx==0 and newIdx==n-1) or (newIdx==0 and idx==n-1):
		return swap2neighSubGroups(msr, idx, newIdx, reversals, pointOrder)

	p = msr.subRoutes[pointOrder[idx]]
	pRight = msr.subRoutes[pointOrder[(idx+1)%n]]
	pLeft = msr.subRoutes[pointOrder[(idx-1+n)%n]]

	newPLeft = msr.subRoutes[pointOrder[(newIdx-1+n)%n]]
	newPRight = msr.subRoutes[pointOrder[newIdx]]

	if reversals[pointOrder[idx]]:
		p1st = p.getSecondEndPoint()
		p2nd = p.getFirstEndPoint()
	else:
		p1st = p.getFirstEndPoint()
		p2nd = p.getSecondEndPoint()

	#Flip them...
	pRight.isReversed = not pRight.isReversed if reversals[pointOrder[(idx+1)%n]] else pRight.isReversed
	pLeft.isReversed = not pLeft.isReversed if reversals[pointOrder[(idx-1+n)%n]] else pLeft.isReversed
	if newPLeft != pRight and newPLeft != pLeft:
		newPLeft.isReversed = not newPLeft.isReversed if reversals[pointOrder[(newIdx-1+n)%n]] else newPLeft.isReversed
	newPRight.isReversed = not newPRight.isReversed if reversals[pointOrder[newIdx]] else newPRight.isReversed

	distBefore = calcDist(p2nd, pRight.getFirstEndPoint()) + calcDist(p1st, pLeft.getSecondEndPoint()) + calcDist(newPLeft.getSecondEndPoint(), newPRight.getFirstEndPoint())

	distAfter = calcDist(p2nd, newPRight.getFirstEndPoint()) + calcDist(p1st, newPLeft.getSecondEndPoint()) + calcDist(pLeft.getSecondEndPoint(), pRight.getFirstEndPoint())

	# ...and flip them back!
	pRight.isReversed = not pRight.isReversed if reversals[pointOrder[(idx+1)%n]] else pRight.isReversed
	pLeft.isReversed = not pLeft.isReversed if reversals[pointOrder[(idx-1+n)%n]] else pLeft.isReversed
	if newPLeft != pRight and newPLeft != pLeft:
		newPLeft.isReversed = not newPLeft.isReversed if reversals[pointOrder[(newIdx-1+n)%n]] else newPLeft.isReversed
	newPRight.isReversed = not newPRight.isReversed if reversals[pointOrder[newIdx]] else newPRight.isReversed

	tmp = pointOrder[idx]

	DEBUG1 = pointOrder[(newIdx-1+n)%n]
	DEBUG2 = pointOrder[newIdx]

	DEBUG3 = pointOrder[(idx-1+n)%n]
	DEBUG4 = pointOrder[(idx+1)%n]

	pointOrder.insert(newIdx, tmp)

	if newIdx > idx:
		pointOrder.pop(idx)
	elif newIdx == idx:
		assert False # This shouldn't happen!

	else:
		pointOrder.pop(idx+1)


	# Only for debugging
	a = -1
	for i in range(n):
		if pointOrder[i] == tmp:
			a = i
			break
	assert a!= -1
	#assert a == newIdx

	assert pointOrder[(a+1)%n] == DEBUG2
	assert pointOrder[(a-1+n)%n] == DEBUG1

	for i in range(n):
		if pointOrder[i] == DEBUG3:
			assert pointOrder[(i+1)%n] == DEBUG4
			break
		assert i!=n-1
	#End of debugging


	return distAfter - distBefore

def getRandomNeigh(msr, reversals, pointOrder):
	n = len(msr.subRoutes)
	r = random()
	if r < 0.33:
		chosenIdx = randint(0, n-1)

		change = distChangeFromReversal(msr, chosenIdx, reversals, pointOrder)

		return change
	elif r < 1.67: # TODO: NO!
		idx = randint(0, n-1)
		newIdx = randint(0, n-1)
		shouldReverse = random()>0.5
		change = moveOneSubGroup(msr, idx, newIdx, reversals, shouldReverse, pointOrder)

		return change

	else:
		assert False
		line1 = randint(0, n-1)
		line2 = randint(0, n-1)

		isFirstCase = random()<0.5

		change = distChangeFrom2opt(msr, line1, line2, isFirstCase)

		a = pointOrder[line1]
		b = pointOrder[(line1+1)%n]
		c = pointOrder[line2]
		d = pointOrder[(line2+1)%n]

		if isFirstCase:
			pointOrder[line1] = a
			pointOrder[(line2+1)%n] = b
			pointOrder[line2] = c
			pointOrder[(line1+1)%n] = d
		else:
			pointOrder[line1] = a
			pointOrder[line2] = b
			pointOrder[(line2+1)%n] = c
			pointOrder[(line1+1)%n] = d

		distBefore = calcDist(msr.subRoutes[lineIdx1].getSecondEndPoint(), msr.subRoutes[(lineIdx1+1)%len(msr.subRoutes)].getFirstEndPoint()) + calcDist(msr.subRoutes[lineIdx2].getSecondEndPoint(), msr.subRoutes[(lineIdx2+1)%len(msr.subRoutes)].getFirstEndPoint())
		if isFirstCase:
			distAfter = calcDist(msr.subRoutes[lineIdx1].getSecondEndPoint(), msr.subRoutes[(lineIdx2+1)%len(msr.subRoutes)].getFirstEndPoint()) + calcDist(msr.subRoutes[lineIdx2].getSecondEndPoint(), msr.subRoutes[(lineIdx1+1)%len(msr.subRoutes)].getFirstEndPoint())
		else:
			distAfter = calcDist(msr.subRoutes[lineIdx1].getSecondEndPoint(), msr.subRoutes[(lineIdx2)%len(msr.subRoutes)].getFirstEndPoint()) + calcDist(msr.subRoutes[lineIdx2+1].getSecondEndPoint(), msr.subRoutes[(lineIdx1+1)%len(msr.subRoutes)].getFirstEndPoint())

		return change

def changeMSR(msr, reversals, pointOrder, dist):

	# Flip reversals
	for i in range(len(msr.subRoutes)):
		if reversals[pointOrder[i]]:
			msr.subRoutes[pointOrder[i]].isReversed = not msr.subRoutes[pointOrder[i]].isReversed

	msr.connections = pointOrder
	#print("old dist:", msr.externalDist, "new:", dist, "true dist:", msr.calcExternalDist())
	#print("conn:", msr.connections)
	msr.externalDist = dist

	return

def doSanityCheck(subroutes, reversals, pointOrder, distIn):
	dist = 0.0

	for i in range(len(subroutes)):
		if reversals[pointOrder[i]]:
			subroutes[pointOrder[i]].isReversed = not subroutes[pointOrder[i]].isReversed

	for i in range(len(subroutes)-1):
		p1 = subroutes[pointOrder[i]].getSecondEndPoint()
		p2 = subroutes[pointOrder[i+1]].getFirstEndPoint()
		dist += calcDist(p1, p2)

	# and the wrap-around-case.
	p1 = subroutes[pointOrder[-1]].getSecondEndPoint()
	p2 = subroutes[pointOrder[0]].getFirstEndPoint()
	dist += calcDist(p1, p2)

	for i in range(len(subroutes)):
		if reversals[pointOrder[i]]:
			subroutes[pointOrder[i]].isReversed = not subroutes[pointOrder[i]].isReversed

	assert fabs(dist-distIn) < 1.0e-8


def calcOptimalSolFor3(msr):
	n = len(msr.subRoutes)

	assert n == 3

	# Set the first one fixed to 0
	permutations = [[0,1,2], [0,2,1]]

	allReversals = [ [i%2<=0, i%4<=1, i<=2] for i in range(8)]

	bestReversal = None
	bestPointOrder = None

	oldDist = msr.externalDist
	bestDist = inf

	for reversals in allReversals:
		for pointOrder in permutations:

			for i in range(len(msr.subRoutes)):
				if reversals[pointOrder[i]]:
					msr.subRoutes[pointOrder[i]].isReversed = reversals[pointOrder[i]]

			msr.connections = pointOrder

			dist = msr.calcExternalDist()

			if dist < bestDist:
				bestDist = dist
				bestReversal = copy(reversals)
				bestPointOrder = copy(pointOrder)

	assert bestDist < oldDist or fabs(bestDist-oldDist) < 1.0e-8
	assert bestReversal != None
	assert bestPointOrder != None

	for i in range(len(msr.subRoutes)):
		if bestReversal[pointOrder[i]]:
			msr.subRoutes[pointOrder[i]].isReversed = bestReversal[pointOrder[i]]

	msr.connections = bestPointOrder

	msr.externalDist = bestDist

def externalOptimization(msr, startDist, maxIter=30):
	assert fabs(msr.getProperTotalDist() - msr.getTotalDist()) < 1.0e-10
	reversals = [False for sr in msr.subRoutes]
	pointOrder = copy(msr.connections)

	if len(msr.subRoutes) < 4:
		# TODO: find the optimal solution
		calcOptimalSolFor3(msr)
		assert fabs(msr.getProperTotalDist() - msr.getTotalDist()) < 1.0e-8
		return

	def acceptanceProb(newVal, orgVal, temp):
		if newVal < orgVal:
			return 1.0
		else:
			return exp(-(newVal-orgVal) / temp)

	def temperature(x):
		return x

	currentDist = startDist

	#print("start dist:", startDist)
	assert fabs(startDist-msr.calcExternalDist()) < 1.0e-8

	for itr in range(maxIter):
		temp = temperature( float(maxIter-itr) / maxIter )

		newOrder = copy(pointOrder) # TODO: This is super slow.
		newReversal = copy(reversals) # TODO: This is super slow.
		distChange = getRandomNeigh(msr, newReversal, newOrder)

		ap = acceptanceProb(currentDist+distChange, currentDist, temp)
		if ap >= random():
			currentDist += distChange

			reversals = newReversal
			pointOrder = newOrder

			if distChange!=0:
				print(reversals, pointOrder, distChange)

			#print("New improvement:", currentDist, distChange, pointOrder)
			doSanityCheck(msr.subRoutes, reversals, pointOrder, currentDist)

	if currentDist < startDist:
		changeMSR(msr, reversals, pointOrder, currentDist)
		#print("It improved!")

	print(msr.getTotalDist(), msr.getProperTotalDist())
	assert fabs(msr.getProperTotalDist() - msr.getTotalDist()) < 1.0e-8

	return
