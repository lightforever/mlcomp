export class AppSettings {
    public static get API_ENDPOINT(): string {
        return `http://${window.location.hostname}:${window.location.port}/api/`
    }

    public static status_colors = {
        'not_ran': 'gray',
        'queued': 'lightblue',
        'in_progress': 'lime',
        'failed': '#e83217',
        'stopped': '#cb88ea',
        'skipped': 'orange',
        'success': 'green'
    };
    public static log_colors = {
        'debug': 'gray',
        'info': 'lightblue',
        'warning': 'lightyellow',
        'error': 'red'
    };
}
