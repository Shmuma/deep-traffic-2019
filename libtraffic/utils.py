import logging
import coloredlogs


def setup_logging():
    field_styles = coloredlogs.DEFAULT_FIELD_STYLES.update({'levelname': {'color': 'cyan', 'bright': True}})
    coloredlogs.install(level=logging.INFO, fmt="%(asctime)-15s %(levelname)s %(message)s",
                        field_styles=field_styles)
