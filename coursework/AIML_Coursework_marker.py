#!/usr/bin/env python
# coding: utf-8

# # AIML Coursework marker
# 
# <div class="alert alert-block alert-danger"> <b>REMEMBER:</b> You need to make sure you are running this code within the virtual environment you created using 'AIenv'.<br> In Jupyter click on the kernel menu then change-kernel. In VSCode use the kernel selector in the top-right hand corner </div>

# In[10]:


import re
import aiml
import random

def preprocessSingleInput(bot,theInput):
    # run the input through the 'normal' subber- only wortks for a single sentence
    subbed1 = bot._subbers['normal'].sub(theInput).upper()
    subbed2 = re.sub(bot._brain._puncStripRE, " ", subbed1)
    return(subbed2)
    


# # Next cell sets up variables
# - You can change the amount of debugging information printed to screen by setting debug=True
# - you can change the name of your input file to something other than "student.aiml" if you want.
# - **Dont change anything else**

# In[11]:


debug = False
debug2 = False
theAIMLfile = 'student.aiml'
theQuestionsFileName = "coursework-questions-and-responses.txt"
responsesFileName = theAIMLfile[:-5] +"-responses.txt"
feedbackFileName = theAIMLfile[:-5] +"-feedback.txt"
NUMQS =45
NUMCONTEXTQS=3
contextQuestions = [35,42,44]


# # Read the questions and answer from file, then randomise the order

# In[12]:


#declare arrays to hold the questions and answers
questions = []
responses = []

# read the questions and answers in
# Using readline() 
qFile = open(theQuestionsFileName, 'r') 
thisQ = 0
  
while True: 
    # Get next line from file 
    line = qFile.readline() 
    if not line: 
        print("unexpected end of file")
        break
    # should be a question
    elif (line[0] != 'Q' ):
        print("didn't get expected question marker Q")
        break;
    elif ( int(line[1:3]) != thisQ):
        print("question had wrong number")
        break
    else:
        questions.append( line[5:-1])
        if(debug2):
            print("question {} is: {}".format(thisQ,questions[thisQ]))        
        
    line = qFile.readline() # next line should be the corresponding answer
    if not line: 
        print("unexpected end of file")
        break
    elif (line[0] != 'A' ):
        print("didn't get expected answer marker A")
        break;
    elif ( int(line[1:3]) != thisQ):
        print("answer had wrong number")
        break
    else:
        responses.append(line[5:-1])
        if(debug2):
            print("response {} is: {}".format(thisQ,responses[thisQ]))
    
    thisQ += 1
    # then read the empty line separating QnA paits
    line = qFile.readline()
    
    # if line is empty 
    # end of file is reached 
    if not line: 
        break
    if(debug2):
        print("")

qFile.close() 




# shuffle the order of the questions except the **three** context-dependent ones
CQ1 = contextQuestions[0]
CQ2 = contextQuestions[1]
CQ3 = contextQuestions[2]
toremove= [(CQ1 - 1),CQ1,(CQ2 - 1),CQ2,(CQ3 - 1),CQ3]
#print(toremove)
# make a shuffled list with the numbers 1...NUMQs except the ones above in
order = []
for i in range (NUMQS):
    if i not in toremove:
        order.append(i)
random.shuffle(order)

#put the context dependent Qs and precursors back in
order.insert(10,(CQ1 -1))
order.insert(11,CQ1)
order.insert(20,(CQ2-1))
order.insert(21,CQ2)
order.insert(30, (CQ3-1))
order.insert(31,CQ3)
#print(order)
#print ( len(order))
for i in range (NUMQS):
    if i not in order:
        print("{} is missing".format(i))

# check that there are the right number
if (thisQ <NUMQS ):
    print("error, only {} question-answer pairs read".format(thisQ))
elif (len(questions) < NUMQS or len(responses)<NUMQS):
    print("error, somehow the questions or responses have not all be saved")
    if(debug):
        print(" {} questions and {}responses read, thisQ = {}".format(len(questions),len(responses),thisQ))
else: 
    print('{} question-response pairs read for testing your bot'.format(thisQ))


# # Create the chatbot

# In[13]:


# Create Chatbot and read the candidate AIML file
checkBot = aiml.Kernel()
checkBot.verbose(True)


# ## Clear any old categories,  reload the AIML file

# In[14]:


checkBot.resetBrain()
checkBot.learn(theAIMLfile)

# How many categories were correctly read
numCategories = checkBot.numCategories()
print( "After reading your file the bot has {} categories".format(numCategories))
print( "Remember that the bot will overwrite categories with the same pattern, that and topic".format(numCategories))
print('This number should help you fix misformed categories if needed\n')


# ### See how frequently different language constrcuts have been used

# In[15]:


# either figure out how to query the bot categories
## or open the student.aiml file and read it line by line looking for <srai> <set> <star/> and <that>
file2 = open(theAIMLfile,'r')
srai_count = 0
set_count = 0
wildcard_count=0
starslash_count=0
that_count = 0
condition_count= 0

#read through line by line coutning uise of AIML constructs
while(True):
    line = file2.readline()
    if not line:
        break
    if "<srai>" in line:
        srai_count += 1
    if "<set" in line: # just use start - they ar hopefullty defining a name for their variable
        set_count += 1
    if ("*" in line) or ("_" in line) or ("^" in line) or ("#" in line):
        wildcard_count +=1
    if "<star" in line: #just look for start of tag in case they used indexing
        starslash_count += 1
    if "<that" in line: #just look for start of tag in case they used indexing
        that_count +=1
    if "<that" in line: #just look for start of tag in case they used indexing
        condition_count +=1
file2.close()       
        


# # Ask the questions, check and store the responses

# In[16]:




# initialise score
numCorrect = 0
numContextQsCorrect=0
numNoMatch=0
responsesFile = open(responsesFileName,'w')

for q in range (NUMQS):
    thisQ = order[q]
    #get bot's response to question
    botResponse = checkBot.respond(questions[thisQ])
    if(botResponse==""):
        numNoMatch +=1
    responsesFile.write('Q{:2d}: {}\n'.format(thisQ, questions[thisQ]))
    responsesFile.write('Expected response: {}\n'.format(responses[thisQ]))
    responsesFile.write('Your bot response: {}\n'.format(botResponse))
    # check if it matches the required input
    if botResponse == responses[thisQ] :
        #print('question {} answered correctly'.format(thisQ))
        responsesFile.write('*** Question answered correctly\n\n')
        numCorrect +=1
        if thisQ in contextQuestions:
            numContextQsCorrect +=1
    else:
        responsesFile.write('Question answered incorrectly\n\n')
        if(debug):
            theInput = questions[thisQ]
            print('Q{} {}\n gets preprocessed as:{}'.format(thisQ,theInput,preprocessSingleInput(checkBot,theInput)))
            print(' expected :' +responses[thisQ])
            print(' got      :' +botResponse)
            lastThat = checkBot.getPredicate("_outputHistory")

# write final line to log file and exit
responsesFile.write(' In total you got {} questions correct'.format(numCorrect))
responsesFile.close()


# # Calculate final score and feedback

# In[17]:



feedbackFile = open(feedbackFileName,'w')


# calculate final score
finalScore= numCorrect 
# if all questions correct then we start rewarding go solutions
if (numCorrect==NUMQS):
    if (numCategories <10):
        finalScore = 100
    else:
        finalScore = 90 - numCategories

# provide output for DEWIS
feedbackFile.write('<SCORE>{}</SCORE>\n'.format(finalScore))

feedback= "<MESSAGE> "
feedback += "After removing duplicates, your bot used " + str(numCategories) + " categories\n"

# what did the submission get wrong and why?
if(numCorrect< NUMQS):
    feedback += "Your bot answered some question incorrectly. File " + responsesFileName + " has more details\n"
    feedback +="- Common mistakes are typos or extra spaces.\n" 
    if(numNoMatch>0):
        feedback += "For " + str(numNoMatch) +" questions your bot did not have a matching category.\n"
    contextErrors = NUMCONTEXTQS - numContextQsCorrect
    if( contextErrors >0 ):
        feedback+= "Your bot answered incorrectly for " + str(contextErrors) + " questions that require a sense of context.\n"
else: #
    feedback += "Your bot answered every question correctly using " + str(numCategories) + " categories\n"
    if ( srai_count==0  or wildcard_count ==0 or starslash_count==0):
        feedback += "You can improve your score by generalising using srai and wildcards.\n"
    if (set_count==0 or that_count==0):
        feedback += "You can improve your score by remembering context and what the conversation is talking about.\n"
    if(condition_count==0):
        feedback += "You can use <condition> to change behaviour within a category\n"
    if(numCategories <=11):
        feedback += "Congratulations, you have matched Jim's score!"

feedback =  feedback + "</MESSAGE>\n"
feedbackFile.write(feedback)
feedbackFile.close()


# # Uncomment the cell below if you want to run your bot interactively

# In[18]:


#keepgoing= True
#while(keepgoing):
#    nextInput = input("Enter your message >> ")
#    if (nextInput=='bye'):
#       keepgoing= False
#    else:
#        print (checkBot.respond(nextInput))


# In[ ]:




