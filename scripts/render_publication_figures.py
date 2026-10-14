"""Publication typography only, using frozen numerical inputs.

No sampler, bootstrap, coefficient fit, or data writer is called. The one
cross-validation plot lacking archived curve arrays is retained and inspected
in its existing, compatible serif style. Artist data are hashed before and
after styling to prevent presentation changes from changing plotted values.
"""
from pathlib import Path
import hashlib
import json
import runpy
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
import numpy as np
import pandas as pd

QA = ROOT / 'tmp' / 'publication_figure_previews'
QA.mkdir(parents=True, exist_ok=True)
records = []
inputs = {}

def read(name, **kwargs):
    path = ROOT / name
    inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path, **kwargs)

def data_digest(fig):
    digest = hashlib.sha256()
    for ax in fig.axes:
        for line in ax.lines:
            for array in (line.get_xdata(), line.get_ydata()):
                digest.update(repr(np.asarray(array).tolist()).encode())
        for collection in ax.collections:
            digest.update(repr(np.asarray(collection.get_offsets()).tolist()).encode())
            for path in collection.get_paths():
                digest.update(path.vertices.tobytes())
        for patch in ax.patches:
            digest.update(patch.get_path().vertices.tobytes())
    return digest.hexdigest()

original_save = Figure.savefig

def publication_save(fig, path, *args, **kwargs):
    path = Path(path)
    if path.suffix.lower() != '.pdf':
        return original_save(fig, path, *args, **kwargs)
    before = data_digest(fig)
    mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
    for text in fig.findobj(Text):
        text.set_fontfamily('DejaVu Serif')
        if hasattr(text, 'set_math_fontfamily'):
            text.set_math_fontfamily('dejavuserif')
    for ax in fig.axes:
        if not ax.axison:
            continue
        ax.xaxis.label.set_fontsize(9.2)
        ax.yaxis.label.set_fontsize(9.2)
        ax.title.set_fontsize(9.2)
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontsize(7.8)
        ax.xaxis.get_offset_text().set_fontsize(7.8)
        ax.yaxis.get_offset_text().set_fontsize(7.8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_edgecolor('#414141')
        ax.tick_params(width=0.7, length=3.0)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(7.3)
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize(7.3)
    if path.stem == 'solid_cavity_mc_replica_audit':
        # Blue/orange encodes N=108/256 in both crystal reference figures.
        pairs = [('#4C78A8', '#0072B2'), ('#B14A68', '#D55E00')]
        def recolor(value):
            try:
                rgba = np.asarray(mpl.colors.to_rgba_array(value)).copy()
            except (ValueError, TypeError):
                return value
            for old,new in pairs:
                selected = np.all(np.isclose(rgba[:,:3],mpl.colors.to_rgb(old)),axis=1)
                rgba[selected,:3] = mpl.colors.to_rgb(new)
            return rgba if len(rgba) != 1 else tuple(rgba[0])
        for artist in fig.findobj():
            for property_name in ('color','facecolor','edgecolor','markerfacecolor','markeredgecolor'):
                getter = getattr(artist,'get_'+property_name,None)
                setter = getattr(artist,'set_'+property_name,None)
                if getter and setter:
                    try:
                        setter(recolor(getter()))
                    except (ValueError, TypeError):
                        pass
        # Keep the structural cloud unobscured; the legend is shared below.
        ax = fig.axes[2]
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        legend = fig.legend(handles, labels, loc='lower center',
                            bbox_to_anchor=(0.5, -0.025), ncol=4,
                            frameon=False, fontsize=7.3)
        for text in legend.get_texts():
            text.set_fontfamily('DejaVu Serif')
    after = data_digest(fig)
    assert before == after, f'Artist coordinates changed: {path.name}'
    kwargs.setdefault('bbox_inches', 'tight')
    original_save(fig, path, *args, **kwargs)
    original_save(fig, QA / (path.stem + '.png'), dpi=155, bbox_inches='tight')
    records.append({'figure': path.name, 'data_digest_before': before,
                    'data_digest_after': after, 'artist_data_unchanged': True})
    print(path.name, flush=True)

Figure.savefig = publication_save
mpl.rcParams.update({'font.family':'DejaVu Serif', 'font.size':8.7,
                     'mathtext.fontset':'dejavuserif', 'pdf.fonttype':42})

if '--replica-only' in sys.argv:
    import solid_cavity_mc as sm
    import solid_cavity_replica_campaign as replicas
    replicas.make_figure(read(replicas.PROCESSED_PATH),read(replicas.RAW_STRUCTURE_PATH),
        read('data_processed/solid_cavity_mc.csv'),
        read('data_raw/solid_cavity_mc_structure.csv'),read(replicas.COMPARISON_PATH),
        sm.load_solid_coexistence_density(ROOT/'data_raw/fernandez2012_coexistence.csv'),
        ROOT/'figures/solid_cavity_mc_replica_audit.pdf')
    report_path = ROOT/'generated/publication_figure_style.json'
    report = json.loads(report_path.read_text())
    report['figures'] = [r for r in report['figures']
                         if r['figure'] != records[0]['figure']] + records
    report['input_sha256'].update(inputs)
    for name,digest in report['input_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0)

runpy.run_path(str(ROOT/'scripts/draw_cavity_geometry.py'))
plt.close('all')

import grand_reweight as gr
gr.plot_cumulants(read('data_processed/cumulant_test.csv'), ROOT/'figures/cumulant_hierarchy.pdf')
gr.plot_finite_size(read('data_processed/finite_size_cavity.csv'), ROOT/'figures/finite_size_collapse.pdf')
import tonks_validation as tv
tv.plot_data(read('data_processed/tonks_validation.csv'), ROOT/'figures/tonks_validation.pdf')
import cs_benchmark as cs
eos = read('data/eos_benchmark.csv')
cs.make_figure(ROOT/'figures/cs_benchmark.pdf', np.sort(eos['eta'].unique()))
import lattice_pulse_scaling as ps
ps.make_figure(*(read('data_processed/'+name+'.csv') for name in
    ('lattice_peak_center_amplitude','lattice_peak_scaling_fits',
     'lattice_width_hierarchical','lattice_width_hierarchical_fit','lattice_pulse_curves')),
    ROOT/'figures/lattice_pulse_scaling.pdf')
read('data_processed/lattice_scalar_joint_validation.csv')
import lattice_leave_lmax_prediction as lm
cp = ROOT/'data_processed/lattice_leave_lmax_prediction_covariance.npz'
inputs[str(cp.relative_to(ROOT))] = hashlib.sha256(cp.read_bytes()).hexdigest()
with np.load(cp) as archive:
    covariance = {target:archive[f'covariance_{target}'] for target in lm.TARGETS}
lm.make_figure(read('data_processed/lattice_leave_lmax_prediction_curves.csv'),
               read('data_processed/lattice_leave_lmax_prediction_diagnostics.csv'),
               covariance, ROOT/'figures/lattice_leave_lmax_prediction.pdf')
import lattice_soft_cavity_family as soft
inputs[str(soft.EXACT_PLOT_INPUT.relative_to(ROOT))] = hashlib.sha256(soft.EXACT_PLOT_INPUT.read_bytes()).hexdigest()
soft.make_figure(read('data_processed/lattice_soft_cavity_features.csv',dtype={'lambda_label':str}),
                 read('data_processed/lattice_soft_cavity_curves.csv',dtype={'lambda_label':str}),
                 ROOT/'figures/lattice_soft_cavity_family.pdf',soft.archived_exact_plot_inputs())
import lattice_exact_controls as exact
exact.make_figure(read('data_processed/lattice_exact_finite_size.csv'),
                  read('data_processed/lattice_interface_weight_estimate.csv'),
                  ROOT/'figures/lattice_exact_finite_size_control.pdf')
import lattice_pulse_area_sum_rule as area
area.make_figure(read('data_processed/lattice_pulse_area_sum_rule.csv'),
                 read('data_processed/lattice_pulse_area_sum_rule_fits.csv'),
                 ROOT/'figures/lattice_pulse_area_sum_rule.pdf')
import lattice_validation_statistics as vs
vs.make_exact_limit_figure(read('data_processed/lattice_exact_limit_fixed_fits.csv'),
                          read('data_processed/lattice_scalar_joint_validation.csv'),
                          ROOT/'figures/lattice_exact_limit_residuals.pdf')
import cavity_mc as cm
cm.make_figure(read('data_processed/cavity_mc.csv'), ROOT/'figures/cavity_mc_benchmark.pdf')
import phase_conditioned as pc
source = read('data_raw/fernandez2012_coexistence.csv', dtype={'N':str})
pc.plot_equal_weight(read('generated/phase_equal_weight_summary.csv'),source,
                      ROOT/'figures/phase_reweighting.pdf')
import hard_sphere_sensitivity as hs
hs.plot_sensitivity(read('data_processed/hard_sphere_model_sensitivity.csv'),
                     read('data_processed/hard_sphere_model_envelope.csv'),
                     read('data_processed/hard_sphere_model_crossings.csv'),
                     ROOT/'figures/hard_sphere_model_sensitivity.pdf')
import solid_cavity_mc as sm
baseline = read('data_processed/solid_cavity_mc.csv')
structure = read('data_raw/solid_cavity_mc_structure.csv')
sm.make_figure(baseline,read('data_raw/solid_cavity_mc_blocks.csv'),structure,
               ROOT/'figures/solid_cavity_mc_benchmark.pdf')
import solid_cavity_replica_campaign as replicas
replicas.make_figure(read(replicas.PROCESSED_PATH),read(replicas.RAW_STRUCTURE_PATH),
                     baseline,structure,read(replicas.COMPARISON_PATH),
                     sm.load_solid_coexistence_density(ROOT/'data_raw/fernandez2012_coexistence.csv'),
                     ROOT/'figures/solid_cavity_mc_replica_audit.pdf')
for name,digest in inputs.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
report = {'scope':'frozen-data figure typography only; no simulation, bootstrap or refit',
          'figures':records,'input_sha256':inputs,'inputs_unchanged':True,
          'retained_compatible_figure':'lattice_holdout_validation.pdf'}
(ROOT/'generated/publication_figure_style.json').write_text(json.dumps(report,indent=2)+'\n')

# Preserve statistical execution records. Only the separately identified
# presentation render receives a fresh output digest (PDF timestamps change
# on a redraw even when every plotted numerical coordinate is identical).
rendered = {'figures/'+item['figure'] for item in records}
for sidecar in (ROOT/'generated').glob('*_metadata.json'):
    record = json.loads(sidecar.read_text())
    render = record.get('figure_render')
    if not render:
        continue
    affected = set(render.get('output_sha256', {})) & rendered
    if not affected:
        continue
    record.setdefault('figure_render_history', []).append(render.copy())
    render = dict(render)
    render['output_sha256'] = dict(render['output_sha256'])
    for name in affected:
        render['output_sha256'][name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    render.update({
        'command': 'python scripts/render_publication_figures.py',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'style_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'frozen-input presentation render only; original statistical execution preserved',
        'artist_data_unchanged': True,
        'verification': 'generated/publication_figure_style.json',
    })
    record['figure_render'] = render
    sidecar.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
