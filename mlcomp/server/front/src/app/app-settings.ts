export class AppSettings {
    public static get API_ENDPOINT(): string{
        return `http://${window.location.hostname}:4201/api/`
    }
    public static status_colors = {
        'not_ran': 'gray', 'queued': 'lightblue', 'in_progress': 'lime',
        'failed': 'red', 'stopped': 'purple', 'skipped': 'orange', 'success': 'green'
    };
    public static log_colors = {
        'debug': 'gray', 'info': 'lightblue', 'warning': 'lightyellow', 'error': 'red'
    };

    public static size(s: number) {
        if (s < Math.pow(2, 10)) {
            return `${s} byte`;
        }
        if (s < Math.pow(2, 20)) {
            return `${Math.floor(s / 1024)} kbyte`;
        }
        if (s < Math.pow(2, 30)) {
            return `${Math.floor(s / Math.pow(2, 20))} mbyte`;
        }

        return `${(s / Math.pow(2, 30)).toFixed(2)} gbyte`;

    }
}
