import {Component, Input} from '@angular/core';
import {DomSanitizer} from '@angular/platform-browser';
import {MatIconRegistry} from '@angular/material';
import {MessageService} from '../../message.service';
import {TaskService} from '../../task.service';
import {Location} from "@angular/common";
import {ActivatedRoute, Router} from "@angular/router";
import {AppSettings} from "../../app-settings";
import {Paginator} from "../../paginator";
import {TaskFilter} from "../../models";
import {ReportService} from "../../report.service";

@Component({
    selector: 'app-tasks',
    templateUrl: './tasks.component.html',
    styleUrls: ['./tasks.component.css']
})
export class TasksComponent extends Paginator<TasksComponent> {
    displayed_columns: string[] = ['id', 'name', 'created', 'started', 'last_activity',
        'status', 'executor', 'dag', 'computer', 'requirements', 'steps', 'links'];
    @Input() dag: number;
    name: string;
    status: string;
    @Input() report: number;

    constructor(protected service: TaskService, protected location: Location,
                private router: Router, private  route: ActivatedRoute,
                iconRegistry: MatIconRegistry, sanitizer: DomSanitizer,
                private message_service: MessageService,
                private report_service: ReportService
    ) {
        super(service, location);
        iconRegistry.addSvgIcon('stop',
            sanitizer.bypassSecurityTrustResourceUrl('assets/img/stop.svg'));
        iconRegistry.addSvgIcon('report',
            sanitizer.bypassSecurityTrustResourceUrl('assets/img/report.svg'));
    }

    protected _ngOnInit() {
        this.route.queryParams.subscribe(params => {
            if(params['dag']) parseInt(this.dag = params['dag']);
            if(params['status']) parseInt(this.status = params['status']);
        });
    }

    filter_name(name: string) {
        this.name = name;
        this.change.emit();
    }


    status_color(status: string) {
        return AppSettings.status_colors[status];
    }


    get_filter() {
        let res = new TaskFilter();
        res.paginator = super.get_filter();
        res.name = this.name;
        res.dag = this.dag;
        res.status = this.status;
        res.report = this.report;
        return res;
    }

    stop(task) {
        this.service.stop(task.id).subscribe(data => {
            task.status = data.status;
        });
    }

    toogle_report(element: any) {
        let self = this;
        this.service.toogle_report(element.id, this.report, element.report_full).subscribe(data => {
            element.report_full = data.report_full;
            self.report_service.data_updated.emit();
        });
    }

    unfinished(element){
        return ['not_ran', 'in_progress', 'queued'].indexOf(element.status)!=-1;
    }
}
