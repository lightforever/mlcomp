import {environment} from "../environments/environment";

export class AppSettings {
    public static get API_ENDPOINT(): string {
        let port = parseInt(window.location.port);
        if(Number.isNaN(port)){
            port = 80
        }
        if(!environment.production){
            port += 1;
        }
        return `http://${window.location.hostname}:${port}/api/`
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
