import {Component, Input} from '@angular/core';
import {Dag, NameCount, DagFilter} from '../../models';
import {DagService} from '../../dag.service';
import {Location} from '@angular/common';
import {Router, ActivatedRoute} from '@angular/router';
import {DomSanitizer} from '@angular/platform-browser';
import {MatIconRegistry} from '@angular/material';
import {MessageService} from '../../message.service';
import {AppSettings} from "../../app-settings";
import {Paginator} from "../../paginator";
import {ReportService} from "../../report.service";

@Component({
    selector: 'app-dags',
    templateUrl: './dags.component.html',
    styleUrls: ['./dags.component.css']
})
export class DagsComponent extends Paginator<Dag> {

    displayed_columns: string[] = ['id', 'name', 'task_count', 'created', 'started', 'last_activity', 'task_status', 'links'];
    project: number;
    name: string;
    @Input() report: number;

    constructor(protected service: DagService, protected location: Location,
                private router: Router, private  route: ActivatedRoute,
                iconRegistry: MatIconRegistry, sanitizer: DomSanitizer,
                private message_service: MessageService,
                private report_service: ReportService
    ) {
        super(service, location);

        iconRegistry.addSvgIcon('config',
            sanitizer.bypassSecurityTrustResourceUrl('assets/img/config.svg'));
        iconRegistry.addSvgIcon('start',
            sanitizer.bypassSecurityTrustResourceUrl('assets/img/play-button.svg'));
        iconRegistry.addSvgIcon('stop',
            sanitizer.bypassSecurityTrustResourceUrl('assets/img/stop.svg'));
        iconRegistry.addSvgIcon('code',
            sanitizer.bypassSecurityTrustResourceUrl('assets/img/programming-code.svg'));
        iconRegistry.addSvgIcon('delete',
            sanitizer.bypassSecurityTrustResourceUrl('assets/img/delete.svg'));
        iconRegistry.addSvgIcon('graph',
            sanitizer.bypassSecurityTrustResourceUrl('assets/img/network.svg'));
        iconRegistry.addSvgIcon('report',
            sanitizer.bypassSecurityTrustResourceUrl('assets/img/report.svg'));
    }

    get_filter(): any {
        let res = new DagFilter();
        res.paginator = super.get_filter();
        res.name = this.name;
        res.project = this.project;
        res.report = this.report;
        return res;
    }

    protected _ngOnInit() {
        this.route.queryParams
            .subscribe(params => {
                if(params['project'])this.project = parseInt(params['project']);
            });

    }

    color_for_task_status(name: string, count: number) {
        return count > 0 ? AppSettings.status_colors[name] : 'gainsboro'
    }

    status_click(dag: Dag, status: NameCount) {
        this.router.navigate([`/dags/dag-detail/${dag.id}`], {queryParams: {status: status.name}});
    }

    filter_name(name: string) {
        this.name = name;
        this.change.emit();
    }

    stop(element: any) {
        this.service.stop(element.id).subscribe(data => element.task_statuses = data.dag.task_statuses);
    }

    toogle_report(element: any) {
        let self = this;
        this.service.toogle_report(element.id, this.report, element.report_full).subscribe(data => {
            element.report_full = data.report_full;
            self.report_service.data_updated.emit();
        });
    }

    has_unfinished(element: Dag) {
        return element.task_statuses[0].count+element.task_statuses[1].count+element.task_statuses[2].count>0;
    }
}
