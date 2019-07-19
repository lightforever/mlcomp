import {Component, Input} from '@angular/core';
import {DomSanitizer} from '@angular/platform-browser';
import {MatDialog, MatIconRegistry} from '@angular/material';
import {MessageService} from '../../message.service';
import {Location} from "@angular/common";
import {ActivatedRoute, Router} from "@angular/router";
import {AppSettings} from "../../app-settings";
import {Paginator} from "../../paginator";
import {TaskFilter} from "../../models";
import {Helpers} from "../../helpers";
import {ReportService} from "../../report/report.service";
import {TaskService} from "../task.service";
import {ModelAddDialogComponent} from "../../model/model-add-dialog.component";

@Component({
    selector: 'app-tasks',
    templateUrl: './tasks.component.html',
    styleUrls: ['./tasks.component.css']
})
export class TasksComponent extends Paginator<TasksComponent> {
    displayed_columns: string[] = [
        'project',
        'dag',
        'id',
        'name',
        'executor',
        'status',
        'created',
        'started',
        'last_activity',
        'duration',
        'computer',
        'requirements',
        'steps',
        'score',
        'links'
    ];
    @Input() dag: number;
    name: string;
    @Input() report: number;

    project: number;

    filter_hidden: boolean = true;
    filter_applied_text: string;

    created_min: string;
    created_max: string;

    not_ran: boolean;
    queued: boolean;
    in_progress: boolean;
    failed: boolean;
    stopped: boolean;
    skipped: boolean;
    success: boolean;

    dags: any[];
    projects: any[];

    last_activity_min: string;
    last_activity_max: string;

    dags_model: any[];


    constructor(protected service: TaskService,
                protected location: Location,
                protected router: Router,
                protected  route: ActivatedRoute,
                iconRegistry: MatIconRegistry,
                sanitizer: DomSanitizer,
                private message_service: MessageService,
                private report_service: ReportService,
                public model_add_dialog: MatDialog
    ) {
        super(service, location);
        iconRegistry.addSvgIcon('stop',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/stop.svg'));
        iconRegistry.addSvgIcon('report',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/report.svg'));
        iconRegistry.addSvgIcon('model',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/model.svg'));
    }

    protected _ngOnInit() {
        let self = this;
        this.route.queryParams.subscribe(params => {
            if (params['dag']) parseInt(this.dag = params['dag']);
            if (params['status']) {
                this[params['status']] = true;
            }

            self.onchange();
        });

        this.data_updated.subscribe(res => {
            self.projects = res.projects;
            self.projects.splice(0, 0,
                {'id': -1, 'name': 'None'});

            self.dags = res.dags;
            self.dags.splice(0, 0,
                {'id': -1, 'name': 'None'});

            self.dags_model = res.dags_model;
        });
    }

    status_color(status: string) {
        return AppSettings.status_colors[status];
    }


    get_filter() {
        let res = new TaskFilter();
        res.paginator = super.get_filter();
        res.name = this.name;
        res.report = this.report;

        if (this.project != -1) {
            res.project = this.project;
        }
        if (this.dag != -1) {
            res.dag = this.dag;
        }

        res.status = {
            'not_ran': this.not_ran,
            'queued': this.queued,
            'in_progress': this.in_progress,
            'failed': this.failed,
            'stopped': this.stopped,
            'skipped': this.skipped,
            'success': this.success
        };
        res.created_min = Helpers.parse_time(this.created_min);
        res.created_max = Helpers.parse_time(this.created_max);
        res.last_activity_min = Helpers.parse_time(this.last_activity_min);
        res.last_activity_max = Helpers.parse_time(this.last_activity_max);

        return res;
    }

    stop(task) {
        this.service.stop(task.id).subscribe(data => {
            task.status = data.status;
        });
    }


    unfinished(element) {
        return [
            'not_ran',
            'in_progress',
            'queued'].indexOf(element.status) != -1;
    }

    onchange() {
        this.change.emit();

        let filter = this.get_filter();
        let count = 0;
        if (this.dag && this.dag != -1) count += 1;
        if (this.name) count += 1;
        if (this.project && this.project != -1) count += 1;
        if (this.created_min) count += 1;
        if (this.created_max) count += 1;
        if (this.last_activity_min) count += 1;
        if (this.last_activity_max) count += 1;
        for (let k of Object.getOwnPropertyNames(filter.status)) {
            count += filter.status[k] == true ? 1 : 0;
        }
        this.filter_applied_text = count > 0 ? `(${count} applied)` : '';
    }

    not_a_model(element) {
        return element.type != 'train';
    }

    model(element){
        this.model_add_dialog.open(ModelAddDialogComponent, {
            width: '500px', height: '400px',
            data: {
                'dag': null,
                'dags':this.dags_model,
                'task': element.id
            }
        });
    }

     is_report_transparent(element: any) {
        if (this.report) {
            return false;
        }

        return !element.report;
    }

    is_report_transparent_active(element: any) {
        if (!this.report) {
            return false;
        }

        return !element.report_full;
    }

    report_click(element: any) {
        let self = this;
        if (this.report) {
            this.service.toogle_report(
                element.id,
                this.report,
                element.report_full).subscribe(data => {
                element.report_full = data.report_full;
                self.report_service.data_updated.emit();
            });
            return
        }
    }
}
