import {Component, Input, OnInit} from "@angular/core";
import {
    MatDialog,
    MatIconRegistry,
    MatTableDataSource
} from "@angular/material";
import {Task} from "../../models";
import {AppSettings} from "../../app-settings";
import {TaskService} from "../task.service";
import {ReportService} from "../../report/report.service";
import {ModelAddDialogComponent} from "../../model/model-add-dialog.component";
import {DomSanitizer} from "@angular/platform-browser";
import {TaskInfoDialogComponent} from "./task-info-dialog.component";

@Component({
    selector: 'app-task-table',
    templateUrl: './task-table.component.html',
    styleUrls: ['./task-table.component.css']
})
export class TaskTableComponent {
    @Input() dataSource: MatTableDataSource<Task>;
    @Input() report: number;
    @Input() total: number;
    @Input() show_links: boolean = true;

    @Input() dags_model: any[];

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

    constructor(protected service: TaskService,
                private report_service: ReportService,
                public model_add_dialog: MatDialog,
                public task_info_dialog: MatDialog,
                iconRegistry: MatIconRegistry,
                sanitizer: DomSanitizer
    ) {
        iconRegistry.addSvgIcon('stop',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/stop.svg'));
        iconRegistry.addSvgIcon('report',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/report.svg'));
        iconRegistry.addSvgIcon('model',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/model.svg'));
        iconRegistry.addSvgIcon('info',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/info.svg'));
    }

    status_color(status: string) {
        return AppSettings.status_colors[status];
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


    not_a_model(element) {
        return element.type != 'train';
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

    model(element) {
        this.model_add_dialog.open(ModelAddDialogComponent, {
            width: '500px', height: '400px',
            data: {
                'dag': null,
                'dags': this.dags_model,
                'task': element.id
            }
        });
    }

    info(element) {
        this.task_info_dialog.open(TaskInfoDialogComponent, {
            width: '600px', height: '700px',
            data: {
                'id': element.id
            }
        });
    }
}