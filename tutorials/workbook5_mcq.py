import ipywidgets as widgets
import sys
from IPython.display import display
from IPython.display import clear_output
from IPython.display import HTML
def create_multipleChoice_widget(description, options, correct_answer):
    if correct_answer not in options:
        options.append(correct_answer)
    
    correct_answer_index = options.index(correct_answer)
    
    radio_options = [(words, i) for i, words in enumerate(options)]
    alternativ = widgets.RadioButtons(
        options = radio_options,
        description = '',
        disabled = False
    )
    
    description_out = widgets.Output()
    with description_out:
        print(description)
        
    feedback_out = widgets.Output()

    def check_selection(b):
        a = int(alternativ.value)
        if a==correct_answer_index:
            s = '\x1b[6;30;42m' + "Correct." + '\x1b[0m' +"\n" #green color
        else:
            s = '\x1b[5;30;41m' + "Wrong. " + '\x1b[0m' +"\n" #red color
        with feedback_out:
            clear_output()
            print(s)
        return
    
    check = widgets.Button(description="submit")
    check.on_click(check_selection)
    
    
    return widgets.VBox([description_out, alternativ, check, feedback_out])


Q1 = create_multipleChoice_widget('How many clusters do you think you might find in the data?',['1','2','3','4','5','6'],'3')

Q2 = create_multipleChoice_widget('What symbol do you use to specify the marker type so that data in a scatter plot is displayed as upside-down triangles ?',['.','s','v','^','x','+'],'v')


Q3 = create_multipleChoice_widget('How tall is the figure created by the call: fig,ax=plt.subplots(figsize=(10,5)?',['10 inches','5 inches','10 cm','5 cm'],'10 inches')
Q4 = create_multipleChoice_widget('How wide is that figure if the next line of code is: fig.set_size_inches(5, 10)',['10 inches','5 inches','10 cm','5 cm'],'5 inches')