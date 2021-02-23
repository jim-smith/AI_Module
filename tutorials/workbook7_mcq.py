import ipywidgets as widgets
import sys
from IPython.display import display
from IPython.display import clear_output
from IPython.display import HTML


def create_multipleChoice_widgetNEW(description, options, correct_answer):
    if correct_answer not in options:
        options.append(correct_answer)

    correct_answer_index = options.index(correct_answer)

    radio_options = [(words, i) for i, words in enumerate(options)]
    alternativ = widgets.RadioButtons(
        options=radio_options,
        description='',
        disabled=False,
        indent=False,
        align='center',
    )

    description_out = widgets.Output(layout=widgets.Layout(width='auto'))

    with description_out:
        print(description)

    feedback_out = widgets.Output()

    def check_selection(b):
        a = int(alternativ.value)
        if a == correct_answer_index:
            s = '\x1b[6;30;42m' + "correct" + '\x1b[0m' + "\n"
        else:
            s = '\x1b[5;30;41m' + "try again" + '\x1b[0m' + "\n"
        with feedback_out:
            feedback_out.clear_output()
            print(s)
        return

    check = widgets.Button(description="check")
    check.on_click(check_selection)

    return widgets.VBox([description_out,
                         alternativ,
                         widgets.HBox([check]), feedback_out],
                        layout=widgets.Layout(display='flex',
                                              flex_flow='column',
                                              align_items='stretch',
                                              width='auto'))


def create_multipleChoice_widget(description, options, correct_answer):
    if correct_answer not in options:
        options.append(correct_answer)

    correct_answer_index = options.index(correct_answer)

    style = {'description_width': 'initial'}
    radio_options = [(words, i) for i, words in enumerate(options)]
    alternativ = widgets.RadioButtons(
        style=style,
        options=radio_options,
        description='',
        disabled=False
    )

    description_out = widgets.Output()
    with description_out:
        print(description)

    feedback_out = widgets.Output()

    def check_selection(b):
        a = int(alternativ.value)
        if a == correct_answer_index:
            s = '\x1b[6;30;42m' + "Correct." + '\x1b[0m' + "\n"  # green color
        else:
            s = '\x1b[5;30;41m' + "Wrong. " + '\x1b[0m' + "\n"  # red color
        with feedback_out:
            clear_output()
            print(s)
        return

    check = widgets.Button(description="submit")
    check.on_click(check_selection)

    return widgets.VBox([description_out, alternativ, check, feedback_out])


Q1 = create_multipleChoice_widget('If Input1 is 0, and Input2 is 1, and the perceptron makes an error, which weight will NOT be changed?',
                                  ['biasweight', 'weight1', 'weight2'], 'weight1')

Q2 = create_multipleChoice_widget('If  the perceptron makes an error, which weight will always be changed?',
                                  ['biasweight', 'weight1', 'weight2'], 'biasweight')


Q3 = create_multipleChoice_widget(    'If Input1 is 0, and Input2 is 1, and the perceptron outputs the right value, will any weights be changed?',
    ['yes', 'no'], 'no')


Q4 = create_multipleChoice_widget('If Input1 is 1,  and the perceptron outputs 1 when it should output 0, what is the change to weight1?',
                                  ['it is increased', 'it is decreased', ''], 'it is decreased')


Q5 = create_multipleChoice_widget('If Input1 is 1,  and the perceptron outputs 0 when it should output 1, what is the change to weight1?',
                                  ['it is increased', 'it is decreased', ''], 'it is increased')

Q6 = create_multipleChoice_widget(    'Is there a single set of weights that would output the right predictions for the OR problem?',
    ['yes', 'no'], 'no')

Q7 = create_multipleChoice_widget('if a perceptron has learned to correctly predict responses for the OR problem, which one weight can we adjust to make it correctly predict the AND problem?',['biasweight', 'weight1', 'weight2'], 'biasweight')
