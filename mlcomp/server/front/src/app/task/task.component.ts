import {Component, OnInit, ViewChild, EventEmitter, Input, ElementRef} from '@angular/core';
import {MatSort, MatTableDataSource, MatPaginator} from '@angular/material';
import {Observable, of as observableOf, merge} from 'rxjs';
import {catchError} from 'rxjs/operators';
import {map} from 'rxjs/operators';
import {startWith} from 'rxjs/operators';
import {switchMap} from 'rxjs/operators';
import {DomSanitizer} from '@angular/platform-browser';
import {MatIconRegistry} from '@angular/material';
import {MessageService} from '../message.service';
import {Dag, PaginatorRes, Task} from '../models';
import {TaskService} from '../task.service';
import {Location} from "@angular/common";
import {ActivatedRoute, Router} from "@angular/router";

@Component({
    selector: 'app-task',
    templateUrl: './task.component.html',
    styleUrls: ['./task.component.css']
})
export class TaskComponent implements OnInit {
    dataSource: MatTableDataSource<Task> = new MatTableDataSource();

    @ViewChild(MatPaginator) paginator: MatPaginator;
    @ViewChild(MatSort) sort: MatSort;
    change: EventEmitter<any> = new EventEmitter();

    displayed_columns: string[] = ['id', 'name', 'created', 'started', 'last_activity',
        'status', 'executor', 'dag', 'computer', 'requirements', 'steps', 'links'];

    isLoading_results = false;
    status_colors = {
        'not_ran': 'gray', 'queued': 'lightblue', 'in_progress': 'lime',
        'failed': 'red', 'stopped': 'purple', 'skipped': 'orange', 'success': 'green'
    };
    total: number;
    dag: number;

    constructor(private service: TaskService, private location: Location,
                private router: Router, private  route: ActivatedRoute,
                iconRegistry: MatIconRegistry, sanitizer: DomSanitizer,
                private message_service: MessageService
    ) {
        iconRegistry.addSvgIcon('stop',
            sanitizer.bypassSecurityTrustResourceUrl('assets/img/stop.svg'));
    }

    ngOnInit() {
        // If the user changes the sort order, reset back to the first page.
        this.sort.sortChange.subscribe(() => this.paginator.pageIndex = 0);

        this.route.queryParams
            .subscribe(params => {
                this.dag = params['dag'];
            });


        merge(this.sort.sortChange, this.paginator.page, this.change)
            .pipe(
                startWith({}),
                switchMap(() => {
                    this.isLoading_results = true;
                    return this.service.get_tasks(
                        this.sort.active ? this.sort.active : '',
                        this.sort.direction ? this.sort.direction == 'desc' : true,
                        this.paginator.pageIndex,
                        this.paginator.pageSize || 15,
                        this.dataSource.filter,
                        this.dag
                    );
                }),
                map(res => {
                    // Flip flag to show that loading has finished.
                    this.isLoading_results = false;

                    return res;
                }),
                catchError(() => {
                    this.isLoading_results = false;
                    return observableOf(new PaginatorRes<Task>());
                })
            ).subscribe(res => {
            this.dataSource.data = res.data;
            this.total = res.total
        });
    }

    applyFilter(filterValue: string) {
        this.dataSource.filter = filterValue;

        if (this.dataSource.paginator) {
            this.dataSource.paginator.firstPage();
        }

        this.change.emit();
    }


}
