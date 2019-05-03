import {Component, OnInit, ViewChild, EventEmitter, Input, ElementRef} from '@angular/core';
import {Dag, PaginatorRes, NameCount} from '../../models';
import {DagService} from '../../dag.service';
import {MatSort, MatTableDataSource, MatPaginator} from '@angular/material';
import {Observable, of as observableOf, merge} from 'rxjs';
import {catchError} from 'rxjs/operators';
import {map} from 'rxjs/operators';
import {startWith} from 'rxjs/operators';
import {switchMap} from 'rxjs/operators';
import {Location} from '@angular/common';
import {Router, ActivatedRoute} from '@angular/router';
import {DomSanitizer} from '@angular/platform-browser';
import {MatIconRegistry} from '@angular/material';
import {MessageService} from '../../message.service';

@Component({
    selector: 'app-dag',
    templateUrl: './dags.component.html',
    styleUrls: ['./dags.component.css']
})
export class DagsComponent implements OnInit {
    dataSource: MatTableDataSource<Dag> = new MatTableDataSource();

    @ViewChild(MatPaginator) paginator: MatPaginator;
    @ViewChild(MatSort) sort: MatSort;
    change: EventEmitter<any> = new EventEmitter();

    displayed_columns: string[] = ['id', 'name', 'task_count', 'created', 'started', 'last_activity', 'task_status', 'links'];
    isLoading_results = false;
    status_colors = {
        'not_ran': 'gray', 'queued': 'lightblue', 'in_progress': 'lime',
        'failed': 'red', 'stopped': 'purple', 'skipped': 'orange', 'success': 'green'
    };
    project: number;
    total: number;
    private interval: number;

    constructor(private dag_service: DagService, private location: Location,
                private router: Router, private  route: ActivatedRoute,
                iconRegistry: MatIconRegistry, sanitizer: DomSanitizer,
                private message_service: MessageService

    ) {
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

    ngOnDestroy(){
        clearInterval(this.interval);
    }

    ngOnInit() {
        // If the user changes the sort order, reset back to the first page.
        this.sort.sortChange.subscribe(() => this.paginator.pageIndex = 0);

        this.route.queryParams
            .subscribe(params => {
                this.project = params['project'];
            });


        merge(this.sort.sortChange, this.paginator.page, this.change)
            .pipe(
                startWith({}),
                switchMap(() => {
                    this.isLoading_results = true;
                    return this.dag_service.getDags(
                        this.sort.active ? this.sort.active : '',
                        this.sort.direction ? this.sort.direction == 'desc' : true,
                        this.paginator.pageIndex,
                        this.paginator.pageSize || 10,
                        this.dataSource.filter,
                        this.project
                    );
                }),
                map(res => {
                    // Flip flag to show that loading has finished.
                    this.isLoading_results = false;

                    return res;
                }),
                catchError(() => {
                    this.isLoading_results = false;
                    return observableOf(new PaginatorRes<Dag>());
                })
            ).subscribe(res => {
            this.dataSource.data = res.data;
            this.total = res.total
        });

        this.interval = setInterval(() => this.change.emit('event'), 5000);
    }

    applyFilter(filterValue: string) {
        this.dataSource.filter = filterValue;

        if (this.dataSource.paginator) {
            this.dataSource.paginator.firstPage();
        }

        this.change.emit();
    }

    color_for_task_status(name: string, count: number) {
        return count > 0 ? this.status_colors[name] : 'gainsboro'
    }

    status_click(dag
                     :
                     Dag, status
                     :
                     NameCount
    ) {
        this.router.navigate(['/tasks'], {queryParams: {dag: dag.id, status: status.name}});
    }
}
