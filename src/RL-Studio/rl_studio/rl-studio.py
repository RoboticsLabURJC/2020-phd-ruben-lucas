import argparse
import rl_studio.config_loader as config_loader

from rl_studio.agents import TrainerFactory, InferencerFactory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=argparse.FileType("r"),
        required=True,
        default="config/config.yaml",
        help="In /config dir you will find .yaml examples files",
    )

    args = parser.parse_args()
    config_loader.set_config_path(args.file.name)
    config_file = config_loader.load_config()

    if config_file["settings"]["mode"] == "inference":
        inferencer = InferencerFactory(config_file)
        inferencer.main()
    else:
        trainer = TrainerFactory(config_file)
        trainer.main()


if __name__ == "__main__":
    main()