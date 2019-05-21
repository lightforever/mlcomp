import {Component} from '@angular/core';
import {Dag, NameCount, DagFilter} from '../../models';
import {DagService} from '../../dag.service';
import {Location} from '@angular/common';
import {Router, ActivatedRoute} from '@angular/router';
import {DomSanitizer} from '@angular/platform-browser';
import {MatIconRegistry} from '@angular/material';
import {MessageService} from '../../message.service';
import {AppSettings} from "../../app-settings";
import {Paginator} from "../../paginator";

@Component({
    selector: 'app-dag',
    templateUrl: './dags.component.html',
    styleUrls: ['./dags.component.css']
})
export class DagsComponent extends Paginator<Dag> {

    displayed_columns: string[] = ['id', 'name', 'task_count', 'created', 'started', 'last_activity', 'task_status', 'links'];
    project: number;
    name: string;

    constructor(protected service: DagService, protected location: Location,
                private router: Router, private  route: ActivatedRoute,
                iconRegistry: MatIconRegistry, sanitizer: DomSanitizer,
                private message_service: MessageService
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
    }

    get_filter(): any {
        let res = new DagFilter();
        res.paginator = super.get_filter();
        res.name = this.name;
        res.project = this.project;
        return res;
    }

    protected _ngOnInit() {
        this.route.queryParams
            .subscribe(params => {
                this.project = params['project'];
            });

    }

    color_for_task_status(name: string, count: number) {
        return count > 0 ? AppSettings.status_colors[name] : 'gainsboro'
    }

    status_click(dag: Dag, status: NameCount) {
        this.router.navigate(['/tasks'], {queryParams: {dag: dag.id, status: status.name}});
    }

    filter_name(name: string) {
        this.name = name;
        this.change.emit();
    }

    stop(element: any) {
        this.service.stop(element.id).subscribe(data=>element.task_statuses=data.dag.task_statuses);
    }
}
