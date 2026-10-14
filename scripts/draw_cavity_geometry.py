"""Static vector geometry illustration; no simulations or numerical results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

root = Path(__file__).resolve().parent.parent
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Serif',
                     'mathtext.fontset': 'dejavuserif', 'pdf.fonttype': 42})
fig = plt.figure(figsize=(7.2, 5.1))
axes = [fig.add_axes([.025,.40,.465,.56]), fig.add_axes([.515,.40,.465,.56]),
        fig.add_axes([.025,.025,.95,.32])]
points = [(1,1),(2.3,1.3),(3.5,.8),(4.8,1.5),(5.5,2.5),(1.2,3),
          (1.7,4.6),(3.4,4.9),(4.8,4.3),(.8,5.5),(5.4,5.4)]
for i,ax in enumerate(axes[:2]):
    ax.set(xlim=(0,10),ylim=(0,8)); ax.axis('off')
    ax.text(0,7.6,('(a) Open physical chamber','(b) Grand-canonical periodic cell')[i],va='top')
    ax.add_patch(Rectangle((.4,.9),5.5,5.4,fill=False,lw=1.5,
                           linestyle='-' if i==0 else '--',edgecolor='#414141'))
    for x,y in points: ax.plot(x,y+.3,'o',ms=4,color='#236b9a')
    ax.add_patch(Circle((3.1,3.5),.85,facecolor='#ffe3ad',edgecolor='#b46b00',lw=1.2))
    ax.plot(3.1,3.5,'+',color='#6a4200',ms=5)
    ax.annotate('',xy=(3.95,3.5),xytext=(3.1,3.5),arrowprops={'arrowstyle':'->','lw':1})
    ax.text(3.48,3.72,'$R$',fontsize=9)
    ax.text(2.93,3.07,r'$\mathbf{r}_0$',fontsize=9)
    ax.text(.7,6.6,r'Cell $\mathcal{D}_V$: fixed volume $V$')
    ax.text(.7,.28,'Exclusion: no particle centers',fontsize=9)
    ax.add_patch(Rectangle((7,2.8),2.6,1.8,facecolor='#e7f2f8',edgecolor='#236b9a'))
    ax.text(8.3,3.7,'Reservoir\n'+r'$T,\mu$',ha='center',va='center')
    ax.annotate('',xy=(7,3.7),xytext=(5.9,3.7),arrowprops={'arrowstyle':'<->','color':'#236b9a','lw':1.3})
    if i==0:
        ax.plot([5.9,5.9],[3.35,4.05],lw=4,color='white')
        ax.text(7.8,5.5,'Exchange\nconnection',ha='center',fontsize=9)
        ax.text(7.8,1.6,'$N$ fluctuates',ha='center',fontsize=9)
    else:
        ax.text(7.7,5.55,'Insertion /\ndeletion',ha='center',fontsize=9)
        ax.text(7.8,1.6,'$N$ fluctuates',ha='center',fontsize=9)
        ax.annotate('',xy=(5.7,1.1),xytext=(.6,1.1),
                    arrowprops={'arrowstyle':'<->','color':'#777777','lw':.8})
        ax.text(3.15,-.35,'Opposite faces identified; wrapping preserves $N$',
                ha='center',fontsize=8.4)
ax=axes[2]; ax.set(xlim=(0,20),ylim=(0,4)); ax.axis('off')
ax.text(0,3.9,'(c) Observation window within a larger sample',va='top')
ax.add_patch(Rectangle((5.4,.35),8.8,2.65,fill=False,edgecolor='#777777'))
ax.add_patch(Rectangle((8,.8),3.3,1.7,fill=False,ls='--',edgecolor='#236b9a',lw=1.2))
for x,y in [(1,1),(2,2.2),(3.5,1.2),(4.7,2),(5.6,1.1),(7,2.3),(8.1,1.1)]:
    ax.plot(x+5,y,'o',ms=4,color='#236b9a')
ax.text(9.65,2.65,'Observation window',ha='center',fontsize=9)
fig.savefig(root/'figures/cavity_geometry_schematic.pdf')
preview = root/'tmp'/'cavity_geometry_schematic.png'
preview.parent.mkdir(exist_ok=True)
fig.savefig(preview,dpi=180)
plt.close(fig)
