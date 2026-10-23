"""Stage self-contained HR2000 model directories locally; no hardware or upload."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import yaml
import zipfile

from pneu_rl.deployment import load_deployment_config
from pneu_utils.utils import get_pkg_path


def archive_models(output):
    archive = output / 'lib3_HR_type1_type2_epi2000_real_ready.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted((output / 'models').glob('*/*')):
            z.write(path, str(Path('pneu_rl') / path.relative_to(output)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    return archive


def main():
    root = Path(get_pkg_path('pneu_rl')).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=root / 'pneu_rl/exp/lib3_real_ready_20261023')
    args = parser.parse_args()
    reports = []
    for kind in (1, 2):
        name = f'261023_lib3_opening_type{kind}_HR_smooth_seed0'
        source = root / 'pneu_rl/models' / name
        dest = args.output / 'models' / name
        cfg = load_deployment_config(name)
        bank = Path(cfg['reference']['kwargs']['trajectory_npz'])
        inputs = {'model.pth': source / 'model_epi_2000.pth',
                  'infos.pkl': source / 'infos.pkl', 'reference.npz': bank}
        for p in inputs.values():
            if not p.is_file():
                raise FileNotFoundError(p)
        if dest.exists():
            raise FileExistsError(f'Refusing to overwrite existing bundle: {dest}')
        dest.mkdir(parents=True)
        hashes = {}
        for filename, path in inputs.items():
            shutil.copy2(path, dest / filename)
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert hashlib.sha256((dest / filename).read_bytes()).hexdigest() == hashes[filename]
        cfg = copy.deepcopy(cfg)
        cfg['reference']['kwargs']['trajectory_npz'] = 'reference.npz'
        cfg['deployment'] = dict(source_model=name, checkpoint_episode=2000,
                                 purpose='real_zero_shot_and_fresh_replay_finetuning',
                                 file_sha256=hashes)
        (dest / 'cfg.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        loaded = load_deployment_config(name, args.output / 'models')
        assert Path(loaded['reference']['kwargs']['trajectory_npz']) == (dest / 'reference.npz').resolve()
        reports.append(dict(model=name, directory=str(dest), sha256=hashes))
    (args.output / 'manifest.json').write_text(json.dumps(reports, indent=2) + '\n')
    archive_models(args.output)
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
