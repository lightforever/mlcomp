import {Component} from '@angular/core';
import {DomSanitizer} from '@angular/platform-browser';
import {MatIconRegistry} from '@angular/material';
import {MessageService} from '../../message.service';
import {TaskService} from '../../task.service';
import {Location} from "@angular/common";
import {ActivatedRoute, Router} from "@angular/router";
import {AppSettings} from "../../app-settings";
import {Paginator} from "../../paginator";
import {TaskFilter} from "../../models";

@Component({
    selector: 'app-task',
    templateUrl: './tasks.component.html',
    styleUrls: ['./tasks.component.css']
})
export class TasksComponent extends Paginator<TasksComponent>{
    displayed_columns: string[] = ['id', 'name', 'created', 'started', 'last_activity',
        'status', 'executor', 'dag', 'computer', 'requirements', 'steps', 'links'];

    total: number;
    dag: number;
    name: string;
    status: string;

    constructor(protected service: TaskService, protected location: Location,
                private router: Router, private  route: ActivatedRoute,
                iconRegistry: MatIconRegistry, sanitizer: DomSanitizer,
                private message_service: MessageService
    ) {
        super(service, location);
        iconRegistry.addSvgIcon('stop',
            sanitizer.bypassSecurityTrustResourceUrl('assets/img/stop.svg'));
    }

    protected _ngOnInit() {
        this.route.queryParams.subscribe(params => {
            this.dag = params['dag'];
            this.status=params['status']
        });
    }

    filter_name(name: string) {
        this.name = name;
        this.change.emit();
    }


    status_color(status: string) {
        return AppSettings.status_colors[status];
    }


    get_filter(){
        let res = new TaskFilter();
        res.paginator = super.get_filter();
        res.name = this.name;
        res.dag = this.dag;
        res.status = this.status;
        return res;
    }

    stop(task) {
        this.service.stop(task.id).subscribe(data=>{
            task.status = data.status;
        });
    }
}
