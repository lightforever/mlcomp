import {Component, Input} from '@angular/core';
import {Dag, NameCount, DagFilter} from '../../models';
import {DagService} from '../dag.service';
import {Location} from '@angular/common';
import {Router, ActivatedRoute} from '@angular/router';
import {DomSanitizer} from '@angular/platform-browser';
import {MatIconRegistry, MatDialog} from '@angular/material';
import {AppSettings} from "../../app-settings";
import {Paginator} from "../../paginator";
import {Helpers} from "../../helpers";
import {DialogComponent} from "../../dialog/dialog.component";

@Component({
    selector: 'app-dags',
    templateUrl: './dags.component.html',
    styleUrls: ['./dags.component.css']
})
export class DagsComponent extends Paginator<Dag> {

    displayed_columns: string[] = [
        'project',
        'id',
        'name',
        'task_count',
        'created',
        'started',
        'last_activity',
        'duration',
        'task_status',
        'links',
        'img_size',
        'file_size'];
    dag: number;
    project: number;
    name: string;
    id_min: number;
    id_max: number;

    @Input() report: number;

    filter_hidden: boolean = true;
    filter_applied_text: string;

    projects: any[];

    created_min: string;
    created_max: string;

    not_ran: boolean;
    queued: boolean;
    in_progress: boolean;
    failed: boolean;
    stopped: boolean;
    skipped: boolean;
    finished: boolean;

    last_activity_min: string;
    last_activity_max: string;

    constructor(protected service: DagService,
                protected location: Location,
                protected router: Router,
                protected  route: ActivatedRoute,
                iconRegistry: MatIconRegistry,
                sanitizer: DomSanitizer,
                public dialog: MatDialog
    ) {
        super(service, location);

        iconRegistry.addSvgIcon('config',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/config.svg'));
        iconRegistry.addSvgIcon('start',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/play-button.svg'));
        iconRegistry.addSvgIcon('stop',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/stop.svg'));
        iconRegistry.addSvgIcon('code',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/programming-code.svg'));
        iconRegistry.addSvgIcon('delete',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/delete.svg'));
        iconRegistry.addSvgIcon('graph',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/network.svg'));
        iconRegistry.addSvgIcon('report',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/report.svg'));
        iconRegistry.addSvgIcon('remove',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/trash.svg'));
        iconRegistry.addSvgIcon('start',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/play-button.svg'));
    }

    get_filter(): any {
        let res = new DagFilter();
        res.paginator = super.get_filter();
        res.name = this.name;
        if (this.project != -1) {
            res.project = this.project;
        }
        res.report = this.report;
        res.status = {
            'not_ran': this.not_ran,
            'queued': this.queued,
            'in_progress': this.in_progress,
            'failed': this.failed,
            'stopped': this.stopped,
            'skipped': this.skipped,
            'finished': this.finished
        };
        res.created_min = Helpers.parse_time(this.created_min);
        res.created_max = Helpers.parse_time(this.created_max);
        res.last_activity_min = Helpers.parse_time(this.last_activity_min);
        res.last_activity_max = Helpers.parse_time(this.last_activity_max);
        res.id_min = this.id_min;
        res.id_max = this.id_max;

        return res;
    }

    protected _ngOnInit() {
        let self = this;
        this.route.queryParams
            .subscribe(params => {
                if (params['project']) {
                    this.project = parseInt(params['project']);
                }
                self.onchange();
            });

        this.data_updated.subscribe(res => {
            if (!res || !res.projects) {
                return;
            }
            self.projects = res.projects;
            self.projects.splice(0, 0,
                {'id': -1, 'name': 'None'}
            );
        });

    }

    color_for_task_status(name: string, count: number) {
        return count > 0 ? AppSettings.status_colors[name] : 'gainsboro'
    }

    status_click(dag: Dag, status: NameCount) {
        this.router.navigate([`/dags/dag-detail/${dag.id}/tasks`],
            {queryParams: {status: status.name}});
    }

    start(element: any) {
        if (!this.can_start(element)) {
            return;
        }
        this.service.start(element.id).subscribe(data => {
            this.change.emit();
        });
    }

    stop(element: any) {
        if (element.success) {
            return;
        }
        if (!this.has_unfinished(element)) {
            return;
        }
        this.service.stop(element.id).subscribe(data =>
            element.task_statuses = data.dag.task_statuses);
    }


    can_start(element: Dag) {
        if (this.has_unfinished(element)) {
            return false;
        }

        return element.task_statuses[3].count +
            element.task_statuses[4].count +
            element.task_statuses[5].count > 0
    }

    has_unfinished(element: Dag) {
        return element.task_statuses[0].count +
            element.task_statuses[1].count +
            element.task_statuses[2].count > 0;
    }

    remove(element: Dag) {
        let self = this;
        if (!element.finished) {
            this.service.stop(element.id).subscribe(data => {
                const dialogRef = this.dialog.open(DialogComponent, {
                    width: '550px', height: '200px',
                    data: {
                        'message': 'The all content will be deleted. ' +
                            'Do you want to continue?'
                    }
                });

                dialogRef.afterClosed().subscribe(result => {
                    if (result) {
                        this.service.remove(element.id).subscribe(res => {
                            self.change.emit();
                        });
                    }
                })

            });
            return;
        }

        const dialogRef = this.dialog.open(DialogComponent, {
            width: '550px', height: '200px',
            data: {
                'message': 'The all content will be deleted. ' +
                    'Do you want to continue?'
            }
        });

        dialogRef.afterClosed().subscribe(result => {
            if (result) {
                this.service.remove(element.id).subscribe(res => {
                    self.change.emit();
                });
            }
        })

    }

    remove_imgs(element: Dag) {
        this.service.remove_imgs(element.id).subscribe(data => {
            element.img_size = 0
        });
    }

    remove_files(element: Dag) {
        if (this.has_unfinished(element)) {
            return;
        }
        this.service.remove_files(element.id).subscribe(data => {
            element.file_size = 0
        });
    }

    size(s: number) {
        return Helpers.size(s);
    }

    onchange() {
        this.change.emit();
        let filter = this.get_filter();
        let count = 0;
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
        if (this.report) {
            this.service.toogle_report(
                element.id,
                this.report,
                element.report_full).subscribe(data => {
                element.report_full = data.report_full;
            });
            return
        }
    }

    report_link(element: any) {
        if (this.report) {
            return null;
        }

        return `/reports/report-detail/${element.report}`;
    }
}
