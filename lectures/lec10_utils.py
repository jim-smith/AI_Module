import matplotlib.pyplot as plt
import copy
import numpy as np
from time import sleep
from IPython.display import clear_output,display




#==========================================================
class Maze: # this assumes maze is rectangular
    # TODO add bounds checking for valid ids
    def __init__(self):
        self.contents = []
        self.width = 0
        self.height = 0
        self.start = 0
        self.goal = 0
        
    def loadFromTxt(self,filename):
        file = open(filename, 'r')
        for line in file.readlines():
            row = []
            for c in line:
                if(c.isspace() and (c!="\n")):
                    row.append(1)
                elif(c!="\n"):
                    row.append(0)
            self.contents.append(row)
        self.height = len(self.contents)
        self.width = len(self.contents[0])
        self.lastColumnId = self.width -1

                        
    def showMaze(self,cmap="Set1"):
        green = 0.3
        yellow = 0.65
    
        #colour start and end point
        self.colourCellFromId( self.start,green)
        self.colourCellFromId( self.goal,yellow)
        
        plt.figure(figsize=(5,5))
        plt.imshow(self.contents,cmap=cmap,norm=None)
              
        
    def setStart(self,x,y):
        self.start = y + self.lastColumnId*x
        
    def setGoal(self,x,y):
        self.goal = y + self.lastColumnId*x    

    def cellidToCoords(self,cellid):
        y = cellid% (self.width -1)
        x = int(cellid/(self.lastColumnId))
        return x,y
    
    def coordsToCellid(self,x,y):
        cellid = y+x*(self.lastColumnId)
        return cellid
    
    def colourCellFromId(self,cellid,colour):
        x,y = self.cellidToCoords(cellid)
        self.contents[x][y] = colour


#======================================================
def setUpMaze(mazefilename):
    maze = Maze()
    maze.loadFromTxt("data/maze.txt")
    maze.setStart(0,9)
    maze.setGoal(20,11)

    maze.showMaze()


    #define the amount to add to the previous cellid for each move
    # can only do this once the maze has been read in so we know how big it is!
    leftMove = -1
    rightMove = 1
    upMove = - (maze.lastColumnId)
    downMove = (maze.lastColumnId)
    # define the set of move so we can iterate through them
    moveSet = [leftMove,rightMove, upMove,downMove]

    return maze,moveSet


#======================================================
class CandidateSolution:
    def __init__(self):
        self.variableValues = []
        self.quality = 0

#======================================================        
 
#python 3 lets us define the types of parameters if we want to
def IsAtGoal(soln:CandidateSolution, theMaze:Maze): 
    lastCell = soln.variableValues [ len(soln.variableValues) -1]
    if (lastCell== theMaze.goal) :
        return True
    else:
        return False
    
    
#======================================================           
def evaluate(solution:CandidateSolution,maze:Maze):
    reason = ""
    #we only need to look at the last position for checking
    position = solution.variableValues[-1]
    
    if(len(solution.variableValues)>1):
        lastposition = solution.variableValues[-2]
        xold,yold = maze.cellidToCoords(lastposition)

    #check is in the maze
    xnew,ynew = maze.cellidToCoords(position)
    if ((xnew < 0) or (xnew> maze.lastColumnId) or (ynew < 0) or (ynew> (maze.height -1))):
        reason = "move takes route out of the maze"
        solution.quality = -1
        
    # and isn;t a wall- which are coded as zero
    elif (maze.contents[xnew][ynew] == 0):
        reason = "move from {},{} to {},{} takes route through wall".format(xold,yold,xnew,ynew)
        solution.quality = -1

    # and isn't going backwards
    elif( len(solution.variableValues)>2 and position==solution.variableValues[-3]):
        reason = "move goes back on itself"
        solution.quality=-1
        
    else: # valid move
        #get coords of goal
        x2,y2 = maze.cellidToCoords(maze.goal)

        #calculate manhattan distance from pythagoras theorem
        euclideanDistance = np.sqrt( (xnew - x2)*(xnew-x2) + (ynew-y2)*(ynew-y2))
        manhattanDistance = np.abs(xnew-x2) + np.abs(ynew-y2)
    
        solution.quality =  manhattanDistance
        
    
    return reason    
   
    
    
    
    
#======================================================           
def displaySearchState(theMaze:Maze, current:CandidateSolution, openList,algname,steps):
    # make a copy of the maze so we can colour in the paths
    newmaze = copy.deepcopy(theMaze)
 
    #set up the colour scheme
    cmap = "Set1"
    green = 0.3
    yellow = 0.65
    blue = 0.2
    orange = 0.5
    
    startx,starty = newmaze.cellidToCoords(newmaze.start)
    endx,endy = newmaze.cellidToCoords(newmaze.goal)
    
    #colour start and end point
    newmaze.colourCellFromId( newmaze.start,green)
    newmaze.colourCellFromId( newmaze.goal,yellow)
    
    # put the path on the current solution in orange
    for position in current.variableValues:
        newmaze.colourCellFromId(position,orange)
 
    
    # put the endpoints of each partial solution in the openlist in blue
    for item in openList:
        lastpos = item.variableValues[-1]
        newmaze.colourCellFromId(lastpos,blue)

    #leavethe old picture on screen for long enpugh to see then refresh
    sleep(0.0075)
    clear_output(wait=True)
    plt.figure(figsize = (7.5,7.5))
    title = "progress for " + algname + " after testing" + str(steps) + " solutions."
    title = title + "\n Current working candidate in orange.\n"
    title = title + "Blue cells indicate solutions on openList"
    plt.title(title)
    plt.axis('off')
    plt.imshow(newmaze.contents,cmap="Set1")
    plt.show()  
    #display(fig)
    
    


