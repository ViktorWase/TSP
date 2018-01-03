from random import random, randint
from math import sqrt, exp, fabs
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


def moveOneSubGroup(msr, idx, newIdx, shouldReverse, pointOrder):
	n = len(msr.subRoutes)
	p = msr.subRoutes[pointOrder[idx]]
	pRight = msr.subRoutes[pointOrder[(idx+1)%n]]
	pLeft = msr.subRoutes[pointOrder[(idx-1+n)%n]]

	newPLeft = msr.subRoutes[pointOrder[(newIdx-1+n)%n]]
	newPRight = msr.subRoutes[pointOrder[newIdx]]

	distBefore = calcDist(p.getSecondEndPoint(), pRight.getFirstEndPoint()) + calcDist(p.getFirstEndPoint(), pLeft.getSecondEndPoint()) + calcDist(newPLeft.getSecondEndPoint(), newPRight.getFirstEndPoint())

	distAfter = calcDist(p.getSecondEndPoint(), newPRight.getFirstEndPoint()) + calcDist(p.getFirstEndPoint(), newPLeft.getSecondEndPoint()) + calcDist(pLeft.getSecondEndPoint(), pRight.getFirstEndPoint())

	tmp = pointOrder[idx]

	pointOrder.insert(newIdx, tmp)

	if newIdx >= idx:
		pointOrder.pop(idx)
	else:
		pointOrder.pop(idx-1)

	return distAfter - distBefore


def getRandomNeigh(msr, reversals, pointOrder):
	n = len(msr.subRoutes)
	r = random()
	if r < 1.33: #TODO: No
		chosenIdx = randint(0, n-1)

		change = distChangeFromReversal(msr, chosenIdx, reversals, pointOrder)

		return change
	elif r < 1.67: # TODO: NO!
		assert False
		idx = randint(0, n-1)
		newIdx = randint(0, n-1)
		shouldReverse = random()>0.5
		change = moveOneSubGroup(msr, idx, newIdx, shouldReverse, pointOrder)

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

	#msr.connections = pointOrder
	#print("old dist:", msr.externalDist, "new:", dist, "true dist:", msr.calcExternalDist())
	msr.externalDist = dist

	return


def externalOptimization(msr, startDist, maxIter=500):
	reversals = [False for sr in msr.subRoutes]
	pointOrder = copy(msr.connections)
	def acceptanceProb(newVal, orgVal, temp):
		if newVal < orgVal:
			return 1.0
		else:
			return exp(-(newVal-orgVal)/temp)

	def temperature(x):
		return x

	currentDist = startDist

	for itr in range(maxIter):
		temp = temperature(float(maxIter-itr)/maxIter)

		newOrder = copy(pointOrder) # TODO: This is super slow.
		newReversal = copy(reversals) # TODO: This is super slow.
		distChange = getRandomNeigh(msr, newReversal, newOrder)

		ap = acceptanceProb(currentDist+distChange, currentDist, temp)
		if ap >= random():
			currentDist += distChange


			reversals = newReversal
			pointOrder = newOrder

			#print("New improvement:", currentDist, distChange)

	if currentDist < startDist:
		changeMSR(msr, reversals, pointOrder, currentDist)
		#print("It improved!")

	#print("Done")
	print(msr.getTotalDist(), msr.getProperTotalDist())
	assert fabs(msr.getProperTotalDist() - msr.getTotalDist()) < 1.0e-10

	return


