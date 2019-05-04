export class AppSettings {
    public static API_ENDPOINT:string='http://127.0.0.1:4201';
    public static status_colors = {
        'not_ran': 'gray', 'queued': 'lightblue', 'in_progress': 'lime',
        'failed': 'red', 'stopped': 'purple', 'skipped': 'orange', 'success': 'green'
    };
}
