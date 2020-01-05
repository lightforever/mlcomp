import {Component, Input, OnInit, ViewChild} from "@angular/core";
import {
    MatDialog,
    MatIconRegistry,
    MatPaginator,
    MatSort
} from "@angular/material";
import {AppSettings} from "../../app-settings";
import {TaskService} from "../task.service";
import {ReportService} from "../../report/report.service";
import {ModelAddDialogComponent} from "../../model/model-add-dialog.component";
import {DomSanitizer} from "@angular/platform-browser";
import {TaskInfoDialogComponent} from "./task-info-dialog.component";
import {Paginator} from "../../paginator";

@Component({
    selector: 'app-task-table',
    templateUrl: './task-table.component.html',
    styleUrls: ['./task-table.component.css']
})
export class TaskTableComponent implements OnInit {
    @Input() paginator: Paginator<any>;
    @Input() report: number;
    @Input() show_links: boolean = true;

    @Input() projects: any[];

    @ViewChild(MatPaginator) paginator_view: MatPaginator;
    @ViewChild(MatSort) sort_view: MatSort;


    displayed_columns: string[] = [
        'project',
        'dag',
        'id',
        'name',
        'status',
        'created',
        'started',
        'last_activity',
        'duration',
        'computer',
        'requirements',
        'steps',
        'progress',
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
                'project': element.dag_rel.project.id,
                'projects': this.projects,
                'task': element.id,
                'file': 'best_full',
                'fold': 0
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

    router_link(element) {
        if (this.report) {
            return null;
        }
        return `/reports/report-detail/${element.report}`
    }

    ngOnInit(): void {
        this.paginator.paginator = this.paginator_view;
        this.paginator.sort = this.sort_view;
        this.paginator.init = true;
        this.paginator.ngOnInit();
    }

    pad(num, size) {
        let s = num + "";
        while (s.length < size) s = "0" + s;
        return s;
    }

    get_progress(element) {
        if(!element.loader_name){
            return ''
        }
        let duration_minutes = this.pad(Math.floor(element.epoch_duration / 60).toString(), 2);
        let duration_seconds = this.pad(Math.floor(element.epoch_duration % 60).toString(), 2);

        let epoch_time_remaining_minutes = this.pad(Math.floor(element.epoch_time_remaining / 60).toString(), 2);
        let epoch_time_remaining_seconds = this.pad(Math.floor(element.epoch_time_remaining % 60).toString(), 2);


        let res = `${element.loader_name}: 
        ${element.batch_index}/${element.batch_total} 
        ${duration_minutes}:${duration_seconds}/
        ${epoch_time_remaining_minutes}:${epoch_time_remaining_seconds}`;
        if(element.loss != null){
            res += `, loss=${element.loss}`
        }
        return res;
    }
}