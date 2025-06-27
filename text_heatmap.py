import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages


latex_special_token = ["!@#$%^&*()"]
# latex generation from https://github.com/jiesutd/Text-Attention-Heatmap-Visualization
def generate(text_list, attention_list, latex_file, color='red', rescale_value = False):
    os.makedirs(os.path.dirname(latex_file), exist_ok=True)
    assert(len(text_list) == len(attention_list))
    if rescale_value:
        attention_list = rescale(attention_list)
    word_num = len(text_list)
    text_list = clean_word(text_list)
    with open(latex_file,'w') as f:
        f.write(r'''\documentclass[varwidth]{standalone}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}'''+'\n')
        string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
        for idx in range(word_num):
            string += "\\colorbox{%s!%s}{"%(color, attention_list[idx])+"\\strut " + text_list[idx]+"} "
        string += "\n}}}"
        f.write(string+'\n')
        f.write(r'''\end{CJK*}
\end{document}''')

def rescale(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_min = np.min(the_array)
    rescale = (the_array - the_min)/(the_max-the_min)*100
    return rescale.tolist()


def clean_word(word_list):
    new_word_list = []
    for word in word_list:
        for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
            if latex_sensitive in word:
                word = word.replace(latex_sensitive, '\\'+latex_sensitive)
        new_word_list.append(word)
    return new_word_list

def colorFader(c1, c2, mix=0):
    return (1 - mix) * np.array(c1) + mix * np.array(c2)

def colored(c, text):
    return f"\033[38;2;{int(c[0])};{int(c[1])};{int(c[2])}m{text} \033[0m"

def render_text_heatmap_terminal(text_list, attention_list, output_file=None, color_start=[0, 255, 0], color_end=[255, 0, 0], rescale_value=False, num_pad_end=0):
    """
    Render text heatmap using ANSI color codes for terminal output.
    Can also save to a text file with HTML-like formatting.
    """
    assert(len(text_list) == len(attention_list) + num_pad_end)
    if rescale_value:
        # print(attention_list)
        attention_list = rescale(attention_list)
        # print(attention_list)
    
    normalizer = max(attention_list)
    output = ""
    
    for word, attention in zip(text_list[:len(attention_list)], attention_list):
        color = colorFader(color_start, color_end, mix=attention / normalizer)
        output += colored(color, word)

    # add regular text to end
    for i in range(num_pad_end):
        output += text_list[-i-1]
    
    # Print to terminal
    print(output)
    
    # Save to file if specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
    
    return output

def render_text_heatmap_matplotlib(text_list, attention_list, pdf_file, color='Reds', rescale_value=False):
    os.makedirs(os.path.dirname(pdf_file), exist_ok=True)
    assert(len(text_list) == len(attention_list))
    if rescale_value:
        attention_list = rescale(attention_list)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(text_list)*0.3), 2))
    
    # Create the heatmap using imshow
    attention_array = np.array([attention_list])  # Make it 2D for imshow
    img = ax.imshow(attention_array, cmap=color, aspect='auto', 
                   extent=[0, len(text_list), 0, 1])
    
    # Set x-axis ticks and labels
    ax.set_xticks([i + 0.5 for i in range(len(text_list))])
    ax.set_xticklabels(text_list, rotation=45, ha='right')
    
    # Remove y-axis ticks and labels
    ax.set_yticks([])
    ax.set_ylabel('')
    
    # Add colorbar
    plt.colorbar(img, ax=ax, shrink=0.8)
    
    # Adjust layout and save
    plt.tight_layout()
    with PdfPages(pdf_file) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.savefig(pdf_file.replace(".pdf", ".png"))
    plt.close(fig)

def render_text_heatmap_matplotlib_backup(text_list, attention_list, pdf_file, color='Reds', rescale_value=False):
    """
    Backup matplotlib function using imshow approach.
    """
    os.makedirs(os.path.dirname(pdf_file), exist_ok=True)
    assert(len(text_list) == len(attention_list))
    if rescale_value:
        attention_list = rescale(attention_list)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(text_list)*0.3), 2))
    
    # Create the heatmap using imshow
    attention_array = np.array([attention_list])  # Make it 2D for imshow
    img = ax.imshow(attention_array, cmap=color, aspect='auto', 
                   extent=[0, len(text_list), 0, 1])
    
    # Set x-axis ticks and labels
    ax.set_xticks([i + 0.5 for i in range(len(text_list))])
    ax.set_xticklabels(text_list, rotation=45, ha='right')
    
    # Remove y-axis ticks and labels
    ax.set_yticks([])
    ax.set_ylabel('')
    
    # Add colorbar
    plt.colorbar(img, ax=ax, shrink=0.8)
    
    # Adjust layout and save
    plt.tight_layout()
    with PdfPages(pdf_file) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.savefig(pdf_file.replace(".pdf", ".png"))
    plt.close(fig)

if __name__ == '__main__':
    ## This is a demo:

    sent = '''riverrun, past Eve and Adam's, from swerve of shore to bend
of bay, brings us by a commodius vicus of recirculation back to
Howth Castle and Environs.
    Sir Tristram, violer d'amores, fr'over the short sea, had passen-
core rearrived from North Armorica on this side the scraggy
isthmus of Europe Minor to wielderfight his penisolate war: nor
had topsawyer's rocks by the stream Oconee exaggerated themselse
to Laurens County's gorgios while they went doublin their mumper
all the time: nor avoice from afire bellowsed mishe mishe to
tauftauf thuartpeatrick: not yet, though venissoon after, had a
kidscad buttended a bland old isaac: not yet, though all's fair in
vanessy, were sosie sesthers wroth with twone nathandjoe. Rot a
peck of pa's malt had Jhem or Shen brewed by arclight and rory
end to the regginbrow was to be seen ringsome on the aquaface.
'''
    words = sent.split()
    word_num = len(words)
    attention = [(x+1.)/word_num*100 for x in range(word_num)]
    import random
    random.seed(42)
    random.shuffle(attention)
    
    # Use the new terminal-based colored text function
    print("Terminal-based colored text heatmap:")
    render_text_heatmap_terminal(words, attention, "tex_sources/sample_colored.txt")
    
    # Also save as matplotlib backup if needed
    print("\nMatplotlib-based heatmap (saved as PDF and PNG):")
    render_text_heatmap_matplotlib_backup(words, attention, "tex_sources/sample_backup.pdf", 'Reds')