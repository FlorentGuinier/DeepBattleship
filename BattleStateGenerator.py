import random
import numpy as np
import cv2
import argparse

# Command line arguments
parser = argparse.ArgumentParser(description='Create one or many random BattleShip game state ...')
parser.add_argument('-p', '--print', type=bool, default=False, required=False)
parser.add_argument('-n', '--numInitialStates', type=int, default=50000, required=False)
parser.add_argument('-s', '--numHitToSimulate', type=int, default=50, required=False)
parser.add_argument('-c', '--minimumChanceToHitABoat', type=float, default=0.33, required=False)
args = parser.parse_args()
shouldPrint = args.print
numInitialStates = args.numInitialStates
numHitToSimulate = args.numHitToSimulate
minimumChanceToHitABoat = args.minimumChanceToHitABoat

# seed numpy for reproductibility
random.seed(0)

# game descriptions:
gridSize = 10
boats = [5, 4, 3, 3, 2]

# each pixel is RGB
# a pixel can be fired at or not yet -> RED channel 255 or 0 respectively
# a pixel can have a boat or not -> BLUE channel 255 or 0 respectively
# GREEN is always 0.
FiredAtChannelIndex = 0
BoatChannelIndex = 2

##################################################################
# helpers
##################################################################
def SaveState(shouldPrint, gameBoardName, gameBoard, numHit):
    if shouldPrint:
        for y in range(gridSize):
            print(gameBoard[:,y,BoatChannelIndex]//255)
    else:
        cv2.imwrite('Data\GameStates\gameBoard-'+gameBoardName+'s'+numHit+'-m'+str(int(minimumChanceToHitABoat*100))+'.png', gameBoard)


def GenerateBoard():
    #zero init game board
    gameBoard = np.zeros((gridSize, gridSize, 3), np.uint8)
    gameBoardName = ""
    #add all boats
    for boatLength in boats:
        boatPlaced = False
        while not boatPlaced:
            numTry = 0
            #find a random position to place boat
            isVertical = random.randint(0, 1)
            yLength = boatLength if isVertical else 1
            xLength = boatLength if not isVertical else 1
            xPos = random.randint(0, gridSize - xLength)
            yPos = random.randint(0, gridSize - yLength)

            #is that position free on the game board?
            posIsFree = True
            for x in range(xPos, xPos+xLength):
                for y in range(yPos, yPos+yLength):
                    if gameBoard[x,y,BoatChannelIndex] != 0:
                        posIsFree = False
                        break
                if not posIsFree:
                    break

            #position is free, add the boat
            if posIsFree:
                gameBoardName = gameBoardName + str(xPos) + str(yPos) + '-'
                boatPlaced = True
                for x in range(xPos, xPos + xLength):
                    for y in range(yPos, yPos + yLength):
                        gameBoard[x, y, BoatChannelIndex] = 255
            else:
                numTry = numTry+1
                if numTry > 5:
                    print('Could not place boat even after retrying 5 or more times ...' + str(numTry))

    SaveState(shouldPrint, gameBoardName, gameBoard, str(0))

    #list all grid pixel with and without boats
    boatPos = []
    noBoatPos = []
    for x in range(gridSize):
        for y in range(gridSize):
            if gameBoard[x, y, BoatChannelIndex] == 255:
                boatPos.append([x,y])
            else:
                noBoatPos.append([x, y])

    for i in range(numHitToSimulate):
        #we want to boost the chance to hit a boat as states were boats are hit are more
        #interresting to be used as training data
        chanceToHitBoat = len(boatPos) / (len(boatPos) + len(noBoatPos))
        chanceToHitBoat = chanceToHitBoat if chanceToHitBoat > minimumChanceToHitABoat else minimumChanceToHitABoat

        if random.random() < chanceToHitBoat:
            pos = boatPos.pop(random.randrange(len(boatPos)))
        else:
            pos = noBoatPos.pop(random.randrange(len(noBoatPos)))
        gameBoard[pos[0], pos[1], FiredAtChannelIndex] = 255

        SaveState(shouldPrint, gameBoardName, gameBoard, str(i+1))

        if len(boatPos) == 0:
            break

for i in range(numInitialStates):
    GenerateBoard()