import logging
import os

def setup_loggers():
    """Setup multiple loggers for different log levels"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Create different loggers for each level
    loggers = {
        'debug': setup_level_logger('debug', 'logs/debug.log', logging.DEBUG, formatter),
        'info': setup_level_logger('info', 'logs/info.log', logging.INFO, formatter),
        'warning': setup_level_logger('warning', 'logs/warning.log', logging.WARNING, formatter),
        'error': setup_level_logger('error', 'logs/error.log', logging.ERROR, formatter),
        'critical': setup_level_logger('critical', 'logs/critical.log', logging.CRITICAL, formatter)
    }
    
    return loggers

def setup_level_logger(name, log_file, level, formatter):
    """Setup individual logger for specific level"""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

# Create logger instances
loggers = setup_loggers()
debug_logger = loggers['debug']
info_logger = loggers['info']
warning_logger = loggers['warning']
error_logger = loggers['error']
critical_logger = loggers['critical']